import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# Configuration
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./routellm_fine_tuned_router"

# Pricing per 1M tokens (Approximate March 2026 rates)
PRICES = {
    "PREMIUM_MODEL": 2.50,   # e.g., GPT-4o
    "BALANCED_MODEL": 0.50,  # e.g., Mixtral 8x7B
    "EFFICIENT_MODEL": 0.05  # e.g., Llama-3-8B
}

def determine_route(score):
    if score >= 4: return "EFFICIENT_MODEL"
    elif score >= 3: return "BALANCED_MODEL"
    else: return "PREMIUM_MODEL"

def evaluate_router():
    # 1. Load Data
    print("Loading test dataset...")
    full_dataset = load_dataset("routellm/gpt4_dataset")
    test_df = full_dataset["test"].to_pandas().sample(n=100, random_state=42) # Start with 100 for speed
    test_df['ground_truth'] = test_df['mixtral_score'].apply(determine_route)

    # 2. Load Model
    print("Loading fine-tuned router...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, local_model_path)
    model.eval()

    # 3. Inference Loop
    results = []
    print(f"Evaluating {len(test_df)} samples...")
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        user_query = row['prompt']
        system_msg = "You are a router model. Respond with: EFFICIENT_MODEL, BALANCED_MODEL, or PREMIUM_MODEL."
        test_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRoute: {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.01, pad_token_id=tokenizer.eos_token_id)
        
        # Decode and clean up prediction
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_output.split("assistant")[-1].strip()
        
        # Guardrail: Ensure prediction is one of the valid labels
        if "EFFICIENT" in prediction: final_pred = "EFFICIENT_MODEL"
        elif "BALANCED" in prediction: final_pred = "BALANCED_MODEL"
        else: final_pred = "PREMIUM_MODEL"
            
        results.append({
            "pred": final_pred,
            "actual": row['ground_truth'],
            "tokens": len(tokenizer.encode(user_query))
        })

    # 4. Calculate Final Metrics
    calculate_token_efficiency(results)

def calculate_token_efficiency(results):
    total_queries = len(results)
    
    # Baseline: If we sent everything to GPT-20B
    baseline_remote_tokens = sum(r['tokens'] for r in results)
    
    # Router results
    actual_remote_tokens = sum(r['tokens'] for r in results if r['pred'] == "PREMIUM_MODEL")
    actual_local_tokens = sum(r['tokens'] for r in results if r['pred'] == "EFFICIENT_MODEL")
    
    # We treat BALANCED_MODEL as Remote for this 2-model comparison
    actual_remote_tokens += sum(r['tokens'] for r in results if r['pred'] == "BALANCED_MODEL")

    token_savings_pct = (actual_local_tokens / baseline_remote_tokens) * 100
    
    print("\n" + "="*40)
    print(f"ROUTER TOKEN EFFICIENCY REPORT")
    print("="*40)
    print(f"Total Queries:            {total_queries}")
    print(f"Total Input Tokens:       {baseline_remote_tokens:,}")
    print(f"Sent to Local (Qwen3-4B): {actual_local_tokens:,}")
    print(f"Sent to Remote (GPT-20B): {actual_remote_tokens:,}")
    print("-" * 40)
    print(f"CLOUD TOKEN REDUCTION:    {token_savings_pct:.2f}%")
    print("="*40)
    
    if token_savings_pct > 50:
        print("RESULT: High Efficiency. Your router is offloading the majority of tasks locally.")
    else:
        print("RESULT: Low Efficiency. Most tasks still require the 20B model.")

if __name__ == "__main__":
    evaluate_router()