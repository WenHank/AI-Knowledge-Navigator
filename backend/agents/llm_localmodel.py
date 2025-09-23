import os
import json
import requests
import torch
import time
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

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
        self.ANSWERING_LOCAL_MODEL_NAME = os.getenv(
            "ANSWERING_LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        
        # Initialize local model
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Timing statistics
        self.timing_stats = {
            "model_load_time": 0.0,
            "total_inference_time": 0.0,
            "inference_count": 0,
            "average_inference_time": 0.0,
            "last_inference_time": 0.0,
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "decoding_time": 0.0,
        }
        
        self._load_local_model()
    
    def _load_local_model(self):
        """Load the local model with error handling and timing"""
        load_start_time = time.time()
        
        try:
            print(f"Loading local model: {self.ANSWERING_LOCAL_MODEL_NAME}")
            
            # Load tokenizer
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.ANSWERING_LOCAL_MODEL_NAME)
            tokenizer_time = time.time() - tokenizer_start
            print(f"⏱️  Tokenizer loaded in {tokenizer_time:.2f}s")
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            model_start = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ANSWERING_LOCAL_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            model_time = time.time() - model_start
            print(f"⏱️  Model loaded in {model_time:.2f}s")
            
            self.model_loaded = True
            total_load_time = time.time() - load_start_time
            self.timing_stats["model_load_time"] = total_load_time
            print(f"✅ Local model loaded successfully in {total_load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("Will use OpenRouter as fallback for all requests")
            self.model_loaded = False
    
    def _generate_local_response(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Generate response using local model with detailed timing"""
        if not self.model_loaded:
            return None
        
        try:
            inference_start_time = time.time()
            
            # Format prompt for instruction-following models
            prompt = f"<s>[INST] {user_query} [/INST]"
            
            # Tokenize input
            tokenization_start = time.time()
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            tokenization_time = time.time() - tokenization_start
            
            # Move to device if using GPU
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            generation_start = time.time()
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
            generation_time = time.time() - generation_start
            
            # Decode response
            decoding_start = time.time()
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            decoding_time = time.time() - decoding_start
            
            total_inference_time = time.time() - inference_start_time
            
            # Update timing statistics
            self.timing_stats["last_inference_time"] = total_inference_time
            self.timing_stats["tokenization_time"] = tokenization_time
            self.timing_stats["generation_time"] = generation_time
            self.timing_stats["decoding_time"] = decoding_time
            self.timing_stats["total_inference_time"] += total_inference_time
            self.timing_stats["inference_count"] += 1
            self.timing_stats["average_inference_time"] = (
                self.timing_stats["total_inference_time"] / self.timing_stats["inference_count"]
            )
            
            # Calculate tokens per second
            input_tokens = len(inputs["input_ids"][0])
            output_tokens = len(outputs[0]) - input_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            print(f"⏱️  Inference completed in {total_inference_time:.2f}s")
            print(f"   └─ Tokenization: {tokenization_time:.3f}s")
            print(f"   └─ Generation: {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            print(f"   └─ Decoding: {decoding_time:.3f}s")
            
            return {
                "response": response.strip(),
                "timing": {
                    "total_time": total_inference_time,
                    "tokenization_time": tokenization_time,
                    "generation_time": generation_time,
                    "decoding_time": decoding_time,
                    "tokens_per_second": tokens_per_second,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            }
            
        except Exception as e:
            print(f"❌ Local model generation failed: {e}")
            return None
    
    def _generate_fallback_response(self, user_query: str) -> dict:
        """Generate response using OpenRouter as fallback with timing"""
        try:
            if not self.OPENROUTER_API_KEY:
                return {"error": "No OpenRouter API key available for fallback"}
            
            print("🔄 Using OpenRouter fallback")
            
            start_time = time.time()
            
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
            
            request_time = time.time() - start_time
            
            if response.status_code == 401:
                return {"error": "OpenRouter authentication failed"}
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                result = {
                    "content": content, 
                    "source": "openrouter_fallback",
                    "timing": {
                        "request_time": request_time
                    }
                }
                
                # Include token usage if available
                if "usage" in response_data:
                    result["token_usage"] = response_data["usage"]
                
                print(f"⏱️  OpenRouter response in {request_time:.2f}s")
                
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
        """Main execution method with local model + fallback logic and timing"""
        user_query = state.get("user_query", "")
        
        if not user_query:
            state["final_answer"] = {"error": "No user query provided"}
            return state
        
        total_start_time = time.time()
        
        # Try local model first
        local_result = self._generate_local_response(user_query)
        
        if local_result is not None:
            print("✅ Local model response generated")
            state["final_answer"] = local_result["response"]
            state["response_source"] = "local_model"
            state["timing"] = local_result["timing"]
            state["timing"]["total_execution_time"] = time.time() - total_start_time
            return state
        
        # Fallback to OpenRouter
        print("🔄 Falling back to OpenRouter")
        fallback_result = self._generate_fallback_response(user_query)
        
        total_execution_time = time.time() - total_start_time
        
        if "error" in fallback_result:
            state["final_answer"] = fallback_result
            state["localagent_error"] = fallback_result["error"]
        else:
            state["final_answer"] = fallback_result["content"]
            state["response_source"] = "openrouter_fallback"
            if "token_usage" in fallback_result:
                state["token_usage"] = fallback_result["token_usage"]
            if "timing" in fallback_result:
                state["timing"] = fallback_result["timing"]
                state["timing"]["total_execution_time"] = total_execution_time
        
        return state
    
    def __call__(self, state: dict) -> dict:
        """Make the agent callable for LangGraph compatibility"""
        return self.run(state)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model including timing stats"""
        info = {
            "local_model_loaded": self.model_loaded,
            "ANSWERING_LOCAL_MODEL_NAME": self.ANSWERING_LOCAL_MODEL_NAME if self.model_loaded else None,
            "fallback_model_name": self.FALLBACK_MODEL_NAME,
            "has_openrouter_key": bool(self.OPENROUTER_API_KEY),
            "device": str(self.model.device) if self.model_loaded else None,
            "timing_statistics": self.timing_stats.copy()
        }
        return info
    
    def reset_timing_stats(self):
        """Reset timing statistics"""
        self.timing_stats = {
            "model_load_time": self.timing_stats["model_load_time"],  # Keep load time
            "total_inference_time": 0.0,
            "inference_count": 0,
            "average_inference_time": 0.0,
            "last_inference_time": 0.0,
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "decoding_time": 0.0,
        }
        print("🔄 Timing statistics reset")
    
    def print_timing_summary(self):
        """Print a summary of timing statistics"""
        stats = self.timing_stats
        print("\n📊 Timing Summary:")
        print(f"   Model Load Time: {stats['model_load_time']:.2f}s")
        print(f"   Total Inferences: {stats['inference_count']}")
        if stats['inference_count'] > 0:
            print(f"   Average Inference: {stats['average_inference_time']:.2f}s")
            print(f"   Last Inference: {stats['last_inference_time']:.2f}s")
            print(f"   Last Breakdown:")
            print(f"     └─ Tokenization: {stats['tokenization_time']:.3f}s")
            print(f"     └─ Generation: {stats['generation_time']:.2f}s")
            print(f"     └─ Decoding: {stats['decoding_time']:.3f}s")


# Example usage and testing
if __name__ == "__main__":
    agent = LocalAgent()
    
    # Print model info
    print("Model Info:", agent.get_model_info())
    
    # Test the agent multiple times to see timing statistics
    test_queries = [
        "介紹你自己",
        "What is artificial intelligence?",
        "Explain machine learning in simple terms."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== Test {i}: {query[:30]}{'...' if len(query) > 30 else ''} ===")
        
        test_state = {
            "user_query": query,
            "execution_summary": {},
        }
        
        result = agent.run(test_state)
        print(f"Response Source: {result.get('response_source')}")
        
        if "timing" in result:
            timing = result["timing"]
            print(f"Execution Time: {timing.get('total_execution_time', 'N/A'):.2f}s")
        
        # Print first 100 chars of response
        answer = str(result.get("final_answer", ""))
        print(f"Response: {answer[:100]}{'...' if len(answer) > 100 else ''}")
    
    # Print timing summary
    agent.print_timing_summary()