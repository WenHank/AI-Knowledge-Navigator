# custom_llm.py
import os
import time
import torch
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class TheCustomLLM(CustomLLM):
    """Custom LLM with local model and OpenRouter fallback"""
    
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.FALLBACK_MODEL_NAME = os.getenv(
            "FALLBACK_MODEL_NAME", "mistralai/mistral-7b-instruct:free"
        )
        self.OPENROUTER_HOST = "https://openrouter.ai/api/v1/chat/completions"
        self.ANSWERING_LOCAL_MODEL_NAME = os.getenv(
            "ANSWERING_LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self._load_local_model()
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            model_name=self.ANSWERING_LOCAL_MODEL_NAME,
        )
    
    def _load_local_model(self):
        try:
            print(f"Loading local model: {self.ANSWERING_LOCAL_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.ANSWERING_LOCAL_MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ANSWERING_LOCAL_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            self.model_loaded = True
            print("Local model loaded successfully")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            print("Will use OpenRouter fallback")
            self.model_loaded = False
    
    def _generate_local_response(self, user_query: str) -> Optional[str]:
        if not self.model_loaded:
            return None
        
        try:
            prompt = f"<s>[INST] {user_query} [/INST]"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
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
            
            response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Local generation failed: {e}")
            return None
    
    def _generate_openrouter_response(self, user_query: str) -> Optional[str]:
        if not self.OPENROUTER_API_KEY:
            return None
        
        try:
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.FALLBACK_MODEL_NAME,
                    "messages": [{"role": "user", "content": user_query}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter failed: {e}")
        return None
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Try local first
        result = self._generate_local_response(prompt)
        if result:
            return CompletionResponse(text=result)
        
        # Fallback to OpenRouter
        result = self._generate_openrouter_response(prompt)
        if result:
            return CompletionResponse(text=result)
        
        return CompletionResponse(text="Unable to process request")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        yield response