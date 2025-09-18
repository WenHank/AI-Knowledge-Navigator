import os
import json
from dotenv import load_dotenv
import requests

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from agents.base import BaseAgent
from agents.prompts import PROCESSING_PROMPT
from agents.extract_json import extract_and_clean_json

# for test it locally
# from base import BaseAgent
# from prompts import PROCESSING_PROMPT
# from extract_json import extract_and_clean_json

load_dotenv()

class FineTunedRouter:
    """Fine-tuned router for local inference"""
    
    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()
        self.system_msg = PROCESSING_PROMPT
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"Loading router from: {self.model_path}")
            
            # Check if path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Read base model path
            base_model_path_file = os.path.join(self.model_path, "base_model_path.txt")
            if os.path.exists(base_model_path_file):
                with open(base_model_path_file, "r", encoding='utf-8') as f:
                    base_model_id = f.read().strip()
            else:
                base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print(f"✅ Router loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading router: {e}")
            self.model = None
            self.tokenizer = None
    
    def route_query(self, query, temperature=0.1, max_new_tokens=20):
        """Route a single query"""
        if self.model is None or self.tokenizer is None:
            print("⚠️  Router not loaded, returning default route")
            return "BALANCED_MODEL", "Model not loaded"
        
        try:
            # Format prompt (same as training)
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRoute: {query.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ],
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"⚠️  Error during routing: {e}")
            return "BALANCED_MODEL", f"Error: {str(e)}"
    


class PreprocessingAgent(BaseAgent):
    def __init__(self, use_local_router=True):
        self.use_local_router = use_local_router
        router_path = "./agents/routellm_fine_tuned_router"
        # Initialize local router if requested
        if self.use_local_router and router_path:
            print("Initializing local fine-tuned router...")
            self.router = FineTunedRouter(router_path)
        else:
            self.router = None
            print("Using OpenRouter API fallback...")
        
        # OpenRouter API configuration (fallback)
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.PREPROCESSING_MODEL_NAME = os.getenv(
            "PREPROCESSING_MODEL_NAME", "meta-llama/llama-3.3-8b-instruct:free"
        )
        self.OPENROUTER_HOST = "https://openrouter.ai/api/v1/chat/completions"
        self.PROCESSING_PROMPT = PROCESSING_PROMPT
    
    def _run_local_router(self, user_query):
        """Use the local fine-tuned router"""
        try:
            # Route the query
            raw_response = self.router.route_query(user_query)

            print(f"📝 Raw Response: {raw_response}")

            # Parse the JSON string into a Python dictionary
            result_dict = json.loads(raw_response)

            # Create JSON response matching your expected format
            result_json = {
                "type": result_dict.get("type", 1),
                "router_decision": result_dict,
            }

            return result_json

        except json.JSONDecodeError as e:
            print(f"❌ Local router error: Failed to parse JSON from response: {e}")
            return {"type": 1, "router_decision": "JSONDecodeError"}
        except Exception as e:
            print(f"❌ Local router error: {e}")
            return {"type": 1, "router_decision": "Exception"}
        
    def _run_openrouter_fallback(self, user_query):
        """Use OpenRouter API as fallback"""
        query = self.PROCESSING_PROMPT + "This is the question: " + user_query
        
        try:
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": self.PREPROCESSING_MODEL_NAME,
                    "messages": [{"role": "user", "content": query}],
                }),
            )
            
            if response.status_code == 401:
                print("Error: Invalid API key or unauthorized access")
                return "1"  # Default fallback
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("🌐 OpenRouter content:", content)
                
                # Process through JSON extractor
                processed_data = extract_and_clean_json(content)
                
                # Extract routing type
                if isinstance(processed_data, dict):
                    routing_type = processed_data.get("type", 1)
                else:
                    try:
                        parsed_data = json.loads(processed_data)
                        routing_type = parsed_data.get("type", 1)
                    except:
                        routing_type = 1
                
                return str(routing_type)
            else:
                print("Error: No choices in API response")
                return "1"
                
        except Exception as e:
            print(f"OpenRouter error: {e}")
            return "1"
    
    def run(self, state: dict) -> dict:
        user_query = state.get("user_query", "")
        
        print(f"🔍 Processing query: {user_query[:100]}...")
        
        # Try local router first if available
        if self.use_local_router and self.router is not None:
            print("Using local fine-tuned router...")
            routing_type = self._run_local_router(user_query)
        else:
            print("Using OpenRouter API fallback...")
            routing_type = self._run_openrouter_fallback(user_query)
        
        print(f"✅ Final routing type: {routing_type}")
        state["preprocessing_result"] = routing_type
        
        return state
    
    def __call__(self, state: dict) -> dict:
        """Make the agent callable for LangGraph compatibility"""
        return self.run(state)


if __name__ == "__main__":
    # Test with different configurations
    
    # Option 1: Use local fine-tuned router
    print("="*60)
    print("TESTING WITH LOCAL FINE-TUNED ROUTER")
    print("="*60)
    
    # Update this path to your actual router location
    router_path = "./routellm_fine_tuned_router"  # or your actual path
    
    try:
        agent = PreprocessingAgent()
        
        # Test with different query types matching your 2-level system
        test_queries = [
            "What is machine learning?",  # Should be type 1 (EASY)
            "What's the capital of France?",  # Should be type 1 (EASY) 
            "How do I make coffee?",  # Should be type 1 (EASY)
            "Explain quantum entanglement's implications for cryptography",  # Should be type 2 (DIFFICULT)
            "Design a distributed system architecture for handling 1M concurrent users",  # Should be type 2 (DIFFICULT)
            "Analyze the geopolitical implications of climate change on global trade patterns",  # Should be type 2 (DIFFICULT)
        ]
        
        for query in test_queries:
            print(f"\n--- Testing: {query[:50]}... ---")
            state = {"user_query": query}
            result_state = agent.run(state)
            print(f"Result type: {result_state['preprocessing_result']}")
            
    except Exception as e:
        print(f"Error with local router: {e}")
        
        # Fallback to OpenRouter
        print("\n" + "="*60)
        print("FALLING BACK TO OPENROUTER API")
        print("="*60)
        