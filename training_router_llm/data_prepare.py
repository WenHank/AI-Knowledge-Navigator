import transformers
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import numpy as np

# Configuration
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./routellm_fine_tuned_router"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Step 1: Load the base model and tokenizer
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix padding side for training

# Load model with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # Use 8-bit quantization
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map={"": 0} if torch.cuda.is_available() else None,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Step 2: Load the RouteML GPT-4 dataset
print("Loading RouteML GPT-4 dataset...")
full_dataset = load_dataset("routellm/gpt4_dataset")
train_df = full_dataset["train"].to_pandas()

print(f"Train dataset size: {len(train_df)}")

# Sample smaller dataset for testing (remove this for full training)
train_df = train_df.sample(n=1000, random_state=42)
print(f"Using sample dataset size: {len(train_df)}")

# Step 3: Define routing strategy
def determine_route(score):
    if score >= 4:
        return "EFFICIENT_MODEL"
    elif score >= 3:
        return "BALANCED_MODEL"
    else:
        return "PREMIUM_MODEL"

train_df['route_decision'] = train_df['mixtral_score'].apply(determine_route)

def format_router_prompt(prompt, route_decision):
    system_msg = "You are a router model. Respond with: EFFICIENT_MODEL, BALANCED_MODEL, or PREMIUM_MODEL."
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRoute: {prompt.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{route_decision}<|eot_id|>"

def tokenize_function(examples):
    prompts = [format_router_prompt(prompt, route) 
              for prompt, route in zip(examples["prompt"], examples["route_decision"])]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=512,  # Reduced for memory
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Step 4: Prepare dataset
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df[['prompt', 'route_decision']])
tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

# Step 5: Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,  # Reduced rank for stability
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./routellm_training_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_strategy="steps",
    eval_strategy="no",
    remove_unused_columns=False,
    report_to="none",
)

# Step 7: Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Step 8: Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation to handle potential gradient issues
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Enable gradient computation
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# Step 9: Train
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    # Try to save what we have so far
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
    print(f"Saved model checkpoint to {local_model_path}")

# Step 10: Save model
print(f"Saving model to {local_model_path}...")
os.makedirs(local_model_path, exist_ok=True)
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)

with open(os.path.join(local_model_path, "base_model_path.txt"), "w") as f:
    f.write(model_id)

print("Training and saving completed!")

# Quick test
def test_model():
    print("\nTesting model...")
    try:
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_8bit=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base_model, local_model_path)
        
        test_prompt = format_router_prompt("What is the capital of France?", "")
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response[-50:]}")
        
    except Exception as e:
        print(f"Test error: {e}")

test_model()