import transformers
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import numpy as np
import evaluate

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Configuration ---
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./routellm_fine_tuned_router"

# Step 1: Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# Step 2: Dataset Loading
full_dataset = load_dataset("routellm/gpt4_dataset")
train_df = full_dataset["train"].to_pandas().sample(n=5000, random_state=42)

# Step 3: Binary Routing Logic (Simplified for Local vs Cloud)
def determine_route(score):
    return "CLOUD_MODEL" if score >= 4.0 else "LOCAL_MODEL"

train_df['route_decision'] = train_df['mixtral_score'].apply(determine_route)

def tokenize_function(examples):
    all_input_ids = []
    all_labels = []
    max_length = 128 

    for prompt, route in zip(examples["prompt"], examples["route_decision"]):
        instruction = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a router. Respond with LOCAL_MODEL or CLOUD_MODEL.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response = f"{route}<|eot_id|>"
        
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        
        input_ids = instruction_ids + response_ids
        label_ids = ([-100] * len(instruction_ids)) + response_ids
        
        # Manual Truncation to be safe
        all_input_ids.append(input_ids[:max_length])
        all_labels.append(label_ids[:max_length])

    return {"input_ids": all_input_ids, "labels": all_labels}

# Step 4: Split and Map
dataset = Dataset.from_pandas(train_df[['prompt', 'route_decision']])
dataset_split = dataset.train_test_split(test_size=0.1)
tokenized_dataset = dataset_split.map(tokenize_function, batched=True, remove_columns=['prompt', 'route_decision'])

# Step 5: LoRA (Increased r for better routing accuracy)
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Accuracy Metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Logits and labels are already shifted by the Trainer for CausalLM loss, 
    # but for accuracy, we usually need to align them manually.
    predictions = np.argmax(logits, axis=-1)
    
    # Flatten and filter for only the answer tokens (labels != -100)
    decoded_preds = predictions.flatten()
    decoded_labels = labels.flatten()
    
    mask = decoded_labels != -100
    return accuracy_metric.compute(predictions=decoded_preds[mask], references=decoded_labels[mask])

from torch.nn.utils.rnn import pad_sequence

def custom_data_collator(features):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]
    
    # Pad input_ids with the tokenizer pad_token_id
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # Pad labels with -100 so they are ignored in loss calculation
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Create the attention mask (1 for real tokens, 0 for padding)
    attention_mask = padded_input_ids.ne(tokenizer.pad_token_id).long()
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

# Step 6 & 7: Args & Collator
training_args = TrainingArguments(
    output_dir="./routellm_training_output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,    # Increased to 32 to keep training stable
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_total_limit=2,
    fp16=True,
    report_to="none",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    # --- CRITICAL NEW SETTINGS ---
    eval_accumulation_steps=1,        # Only keep 1 step of eval in GPU at a time
    prediction_loss_only=True,        # Temporarily disable accuracy to save memory
    remove_unused_columns=False       # Keeps data flow clean
)
# Step 8: Standard Trainer (Simplified)
# Step 8: Updated Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=custom_data_collator,
    # compute_metrics=compute_metrics, # Comment this out to test
)

# Step 9 & 10: Train and Save
trainer.train()
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print(f"Training complete. Model saved to {local_model_path}")