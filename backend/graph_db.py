import os
import time
import torch
import json  # Missing import
from typing import Any, Dict, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings, PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

class YourCustomLLM(CustomLLM):
    """Your existing LLM class with minimal LlamaIndex integration"""
    
    # Define fields that Pydantic can accept
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra attributes
    
    def __init__(self, **kwargs):
        # Call super first to initialize Pydantic properly
        super().__init__(**kwargs)
        
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
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            model_name=getattr(self, 'ANSWERING_LOCAL_MODEL_NAME', self.model_name),
        )
    
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
    
    def _generate_openrouter_response(self, user_query: str) -> Optional[str]:
        """Generate response using OpenRouter as fallback"""
        try:
            if not self.OPENROUTER_API_KEY:
                print("❌ No OpenRouter API key available for fallback")
                return None
            
            print("🔄 Using OpenRouter fallback")
            
            start_time = time.time()
            
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={  # Use json parameter instead of data + json.dumps
                    "model": self.FALLBACK_MODEL_NAME,
                    "messages": [{"role": "user", "content": user_query}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
                timeout=30
            )
            
            request_time = time.time() - start_time
            
            if response.status_code == 401:
                print("❌ OpenRouter authentication failed")
                return None
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print(f"⏱️  OpenRouter response in {request_time:.2f}s")
                return content
            else:
                print("❌ No response content from OpenRouter")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenRouter request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ OpenRouter JSON parsing failed: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ OpenRouter unexpected error: {str(e)}")
            return None
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Main completion method for LlamaIndex"""
        # Try local model first
        local_result = self._generate_local_response(prompt)
        
        if local_result:
            print("✅ Used local model")
            return CompletionResponse(
                text=local_result["response"],
                additional_kwargs=local_result["timing"]
            )
        
        # Fallback to OpenRouter
        print("🔄 Falling back to OpenRouter...")
        openrouter_result = self._generate_openrouter_response(prompt)
        
        if openrouter_result:
            print("✅ Used OpenRouter fallback")
            return CompletionResponse(text=openrouter_result)
        
        # Final fallback
        print("❌ All LLM methods failed")
        return CompletionResponse(text="I'm sorry, I'm unable to process your request at the moment.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Streaming not implemented"""
        response = self.complete(prompt, **kwargs)
        yield response


def create_graphrag_from_markdown_folder(
    markdown_folder_path: str,
    neo4j_config: Optional[Dict] = None
):
    """
    Simple function to create GraphRAG from a folder of markdown files
    
    Args:
        markdown_folder_path: Path to folder containing .md files
        neo4j_config: Optional Neo4j configuration dict
    
    Returns:
        PropertyGraphIndex ready for querying
    """
    
    print("🚀 Initializing GraphRAG with your custom LLM...")
    
    # 1. Initialize your custom LLM (this is all you asked for!)
    llm = YourCustomLLM()
    
    # 2. Set up embedding model (local, no API costs)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. Configure LlamaIndex global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 4. Load markdown files (no chunking needed - GraphRAG handles it)
    print(f"📁 Loading markdown files from: {markdown_folder_path}")
    
    # Check if path is a file or directory
    markdown_path = Path(markdown_folder_path)
    
    if markdown_path.is_file():
        # Single file
        from llama_index.core import Document
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents = [Document(text=content, metadata={"filename": markdown_path.name})]
        print(f"📄 Loaded single markdown file: {markdown_path.name}")
    elif markdown_path.is_dir():
        # Directory of files
        documents = SimpleDirectoryReader(
            input_dir=str(markdown_path),
            file_extractor={".md": "MarkdownReader"},  # Ensure markdown parsing
            recursive=True  # Include subfolders
        ).load_data()
        print(f"📄 Loaded {len(documents)} markdown documents from directory")
    else:
        raise FileNotFoundError(f"Path does not exist: {markdown_folder_path}")
    
    if not documents:
        raise ValueError("No documents were loaded. Check your file path and format.")
    
    # 5. Optional: Set up graph store (Neo4j or in-memory)
    graph_store = None
    if neo4j_config:
        try:
            graph_store = Neo4jPropertyGraphStore(
                username=neo4j_config.get("username", "neo4j"),
                password=neo4j_config.get("password", "password"),
                url=neo4j_config.get("url", "bolt://localhost:7687"),
                database=neo4j_config.get("database", "neo4j")
            )
            print("✅ Connected to Neo4j")
        except Exception as e:
            print(f"⚠️  Neo4j connection failed: {e}")
            print("📝 Using in-memory graph store instead")
    else:
        print("📝 Using in-memory graph store")
    
    # 6. Build PropertyGraph Index (this does the GraphRAG magic)
    print("🔨 Building GraphRAG index from markdown files...")
    print("⚠️  This may take a while depending on document size...")
    
    try:
        if graph_store:
            index = PropertyGraphIndex.from_documents(
                documents,
                property_graph_store=graph_store,
                embed_kg_nodes=True,
                show_progress=True
            )
        else:
            # Simple in-memory version
            index = PropertyGraphIndex.from_documents(
                documents,
                embed_kg_nodes=True,
                show_progress=True
            )
        
        print("✅ GraphRAG index built successfully!")
        print(f"📊 Stats: {llm.timing_stats}")
        
        return index
        
    except Exception as e:
        print(f"❌ Failed to build GraphRAG index: {e}")
        raise


# Example usage
if __name__ == "__main__":

    # Path to your markdown files
    MARKDOWN_PATH = "./output/CV.md"
    
    # Enable Neo4j config
    neo4j_config = {
        "username": "neo4j",
        "password": "password", 
        "url": "bolt://localhost:7687"
    }
    
    try:
        # Create GraphRAG index from markdown files
        index = create_graphrag_from_markdown_folder(
            markdown_folder_path=MARKDOWN_PATH,
            neo4j_config=neo4j_config  # Now using Neo4j!
        )
        
        # Rest of your code...
        
        # Now you can query the system
        query_engine = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize"
        )
        
        # Example queries
        queries = [
            "What are the main topics discussed in the documents?",
            "Summarize the key concepts and their relationships",
            "What entities are mentioned most frequently?"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            try:
                response = query_engine.query(query)
                print(f"📝 Response: {response}")
            except Exception as e:
                print(f"❌ Query failed: {e}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Application failed: {e}")
        import traceback
        traceback.print_exc()