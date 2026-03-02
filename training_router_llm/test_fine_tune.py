import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./routellm_fine_tuned_router"

def test_model():
    print("Loading model for testing...")
    
    # Use 4-bit again to ensure it fits in memory during test
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load your fine-tuned LoRA weights
    model = PeftModel.from_pretrained(base_model, local_model_path)
    model.eval()

    # Define the prompt
    user_query = "What is the capital of France?"
    system_msg = "You are a router model. Respond with: EFFICIENT_MODEL, BALANCED_MODEL, or PREMIUM_MODEL."
    test_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRoute: {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Tokenize and MOVE TO DEVICE
    inputs = tokenizer(test_prompt, return_tensors="pt").to(base_model.device)

    print("\n--- Generating Route Decision ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, 
            temperature=0.1, # Low temp for consistent labels
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's part
    print(f"Full Model Output:\n{response}")

if __name__ == "__main__":
    test_model()