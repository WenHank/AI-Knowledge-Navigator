import os
import json
import requests
import torch
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from agents.base import BaseAgent

load_dotenv()

class LocalAgent(BaseAgent):
    def __init__(self):
        # OpenRouter fallback configuration
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.FALLBACK_MODEL_NAME = os.getenv(
            "FALLBACK_MODEL_NAME", "mistralai/mistral-7b-instruct:free"
        )
        self.OPENROUTER_HOST = "https://openrouter.ai/api/v1/chat/completions"
        
        # Local model configuration
        self.LOCAL_MODEL_NAME = os.getenv(
            "LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        
        # Initialize local model
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        self._load_local_model()
    
    def _load_local_model(self):
        """Load the local model with error handling"""
        try:
            print(f"Loading local model: {self.LOCAL_MODEL_NAME}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_NAME)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.LOCAL_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            self.model_loaded = True
            print("✅ Local model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("Will use OpenRouter as fallback for all requests")
            self.model_loaded = False
    
    def _generate_local_response(self, user_query: str) -> Optional[str]:
        """Generate response using local model"""
        if not self.model_loaded:
            return None
        
        try:
            # Format prompt for instruction-following models
            prompt = f"<s>[INST] {user_query} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Local model generation failed: {e}")
            return None
    
    def _generate_fallback_response(self, user_query: str) -> dict:
        """Generate response using OpenRouter as fallback"""
        try:
            if not self.OPENROUTER_API_KEY:
                return {"error": "No OpenRouter API key available for fallback"}
            
            print("🔄 Using OpenRouter fallback")
            
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": self.FALLBACK_MODEL_NAME,
                    "messages": [{"role": "user", "content": user_query}],
                }),
                timeout=30
            )
            
            if response.status_code == 401:
                return {"error": "OpenRouter authentication failed"}
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                result = {"content": content, "source": "openrouter_fallback"}
                
                # Include token usage if available
                if "usage" in response_data:
                    result["token_usage"] = response_data["usage"]
                
                return result
            else:
                return {"error": "No response content from OpenRouter"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"OpenRouter request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"OpenRouter JSON parsing failed: {str(e)}"}
        except Exception as e:
            return {"error": f"OpenRouter unexpected error: {str(e)}"}
    
    def run(self, state: dict) -> dict:
        """Main execution method with local model + fallback logic"""
        user_query = state.get("user_query", "")
        
        if not user_query:
            state["final_answer"] = {"error": "No user query provided"}
            return state
        
        # Try local model first
        local_response = self._generate_local_response(user_query)
        
        if local_response is not None:
            print("✅ Local model response generated")
            state["final_answer"] = local_response
            state["response_source"] = "local_model"
            return state
        
        # Fallback to OpenRouter
        print("🔄 Falling back to OpenRouter")
        fallback_result = self._generate_fallback_response(user_query)
        
        if "error" in fallback_result:
            state["final_answer"] = fallback_result
            state["localagent_error"] = fallback_result["error"]
        else:
            state["final_answer"] = fallback_result["content"]
            state["response_source"] = "openrouter_fallback"
            if "token_usage" in fallback_result:
                state["token_usage"] = fallback_result["token_usage"]
        
        return state
    
    def __call__(self, state: dict) -> dict:
        """Make the agent callable for LangGraph compatibility"""
        return self.run(state)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "local_model_loaded": self.model_loaded,
            "local_model_name": self.LOCAL_MODEL_NAME if self.model_loaded else None,
            "fallback_model_name": self.FALLBACK_MODEL_NAME,
            "has_openrouter_key": bool(self.OPENROUTER_API_KEY),
            "device": str(self.model.device) if self.model_loaded else None,
        }


# Example usage and testing
if __name__ == "__main__":
    agent = LocalAgent()
    
    # Print model info
    print("Model Info:", agent.get_model_info())
    
    # Test the agent
    test_state = {
        "user_query": "介紹你自己",
        "execution_summary": {},
    }
    
    result = agent.run(test_state)
    print("Response Source:", result.get("response_source"))
    print("Final Answer:", result.get("final_answer")[:200] + "..." if len(str(result.get("final_answer"))) > 200 else result.get("final_answer"))