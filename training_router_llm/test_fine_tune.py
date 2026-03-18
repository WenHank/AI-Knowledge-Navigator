import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_MODEL_PATH = "./routellm_fine_tuned_router"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
model.eval()

def validate_with_llm(model_output, reference_answer):
    """
    Uses the LLM to judge if the Student Answer is semantically 
    equivalent to the Reference Answer.
    """
    # 1. Prepare the grading prompt
    judge_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a strict grading assistant. Compare the Student Answer against the Reference Answer.\n"
        f"If the Student Answer is factually correct and covers the main points, respond with 'YES'.\n"
        f"If the Student Answer is wrong, incomplete, or irrelevant, respond with 'NO'.\n"
        f"Respond ONLY with 'YES' or 'NO'.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Reference Answer: {reference_answer}\n"
        f"Student Answer: {model_output}\n\n"
        f"Is the Student Answer correct?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # 2. Tokenize and Generate
    inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Use a very low temperature (0.01) for consistent grading
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 3. Parse the result
    judgment = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip().upper()
    
    return "YES" in judgment

# ==========================================
# 2. Evaluation Logic
# ==========================================
def evaluate_system_performance(total_samples=500):
    dataset = load_dataset("routellm/gpt4_dataset")["validation"].to_pandas()
    
    # Stratified Sampling
    hard_df = dataset[dataset['mixtral_score'] >= 4.0].sample(int(total_samples/2))
    easy_df = dataset[dataset['mixtral_score'] < 4.0].sample(int(total_samples/2))
    test_df = pd.concat([hard_df, easy_df])

    results = []
    
    # NEW: Pricing for Cost Analysis
    COST_INPUT_1M = 5.00
    COST_OUTPUT_1M = 15.00
    saved_dollars = 0

    print(f"Evaluating {total_samples} samples...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        user_query = row['prompt']
        ref_answer = row['gpt4_response']
        is_actually_easy = row['mixtral_score'] < 4.0
        
        # --- PHASE 1: ROUTING ---
        route_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a specialized Task Router. ONE WORD ONLY: 'EFFICIENT' or 'CLOUD'.\n"
            f"- EFFICIENT: Easy tasks / short answers.\n"
            f"- CLOUD: Hard tasks / complex logic.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Route this: {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        route_inputs = tokenizer(route_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            route_output = model.generate(**route_inputs, max_new_tokens=10)
        
        prediction = tokenizer.decode(route_output[0], skip_special_tokens=True).split("assistant")[-1].strip().upper()
        routed_to_local = "EFFICIENT" in prediction

        # --- PHASE 2: INFERENCE & JUDGING ---
        final_answer = ""
        if routed_to_local:
            # RUN LOCAL MODEL
            ans_inputs = tokenizer(user_query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ans_output = model.generate(**ans_inputs, max_new_tokens=250)
            final_answer = tokenizer.decode(ans_output[0], skip_special_tokens=True)
            
            # Update Savings
            in_tokens = len(ans_inputs[0])
            out_tokens = len(ans_output[0])
            saved_dollars += ((in_tokens/1e6)*COST_INPUT_1M) + ((out_tokens/1e6)*COST_OUTPUT_1M)
        else:
            # USE CLOUD (Reference)
            final_answer = ref_answer

        # JUDGE ACCURACY: Compare final_answer to ref_answer
        # Use your validate_with_llm function here
        is_correct = validate_with_llm(final_answer, ref_answer)

        results.append({
            "is_hard": not is_actually_easy,
            "routed_local": routed_to_local,
            "routing_correct": (routed_to_local == is_actually_easy),
            "output_correct": is_correct
        })

    df_res = pd.DataFrame(results)
    
    print("\n" + "="*55)
    print(f"OVERALL SYSTEM ACCURACY (JUDGE): {df_res['output_correct'].mean()*100:.2f}%")
    print(f"ROUTING ACCURACY: {df_res['routing_correct'].mean()*100:.2f}%")
    print(f"TOTAL MONEY SAVED: ${saved_dollars:.4f}")
    print("="*55)

if __name__ == "__main__":
    evaluate_system_performance(500)