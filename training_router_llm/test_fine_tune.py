import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# Configuration
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./routellm_fine_tuned_router"

def determine_route(score):
    if score >= 4.0:
        return "LOCAL_MODEL"
    else:
        return "CLOUD_MODEL"

def evaluate_router():
    # 1. Load Data
    print("Loading test dataset...")
    full_dataset = load_dataset("routellm/gpt4_dataset")
    test_df = full_dataset["validation"].to_pandas().sample(n=100, random_state=42) # Start with 100 for speed
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
    
    # 1. Calculate Accuracy
    # We need to map the 3-tier predictions to the 2-tier ground truth
    # Logic: EFFICIENT -> LOCAL | BALANCED/PREMIUM -> CLOUD
    correct_predictions = 0
    for r in results:
        mapped_pred = "LOCAL_MODEL" if r['pred'] == "EFFICIENT_MODEL" else "CLOUD_MODEL"
        if mapped_pred == r['actual']:
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_queries) * 100

    # 2. Token Calculations
    total_input_tokens = sum(r['tokens'] for r in results)
    
    # Tokens routed to local (EFFICIENT)
    tokens_routed_local = sum(r['tokens'] for r in results if r['pred'] == "EFFICIENT_MODEL")
    
    # Tokens routed to cloud (BALANCED + PREMIUM)
    tokens_routed_cloud = sum(r['tokens'] for r in results if r['pred'] in ["BALANCED_MODEL", "PREMIUM_MODEL"])
    
    # Total Savings (The tokens that DID NOT go to the cloud)
    tokens_saved = tokens_routed_local
    token_savings_pct = (tokens_saved / total_input_tokens) * 100

    # 3. Print Report
    print("\n" + "="*45)
    print(f"{'ROUTER PERFORMANCE REPORT':^45}")
    print("="*45)
    print(f"Total Queries Evaluated:    {total_queries}")
    print(f"Router Classification Acc:  {accuracy:.2f}%")
    print("-" * 45)
    print(f"Total Potential Tokens:     {total_input_tokens:,}")
    print(f"Tokens Sent to Cloud:       {tokens_routed_cloud:,}")
    print(f"Tokens Routed Locally:      {tokens_routed_local:,}")
    print("-" * 45)
    print(f"NET TOKENS SAVED:           {tokens_saved:,}")
    print(f"CLOUD TOKEN REDUCTION:      {token_savings_pct:.2f}%")
    print("-" * 45)
    print(f"Accuracy:           {accuracy:.2f}%")
    print("="*45)
    
    if accuracy < 70:
        print("NOTE: Accuracy is low. The router might be misclassifying complex tasks.")
    if token_savings_pct > 50:
        print("RESULT: High Efficiency. You are saving significant API costs.")

if __name__ == "__main__":
    evaluate_router()