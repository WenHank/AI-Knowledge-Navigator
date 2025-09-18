# test_gpu.py - Test your GPU setup
import torch
import transformers

def test_gpu_setup():
    print("=== GPU Setup Test ===")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    else:
        print("CUDA not available - will use CPU")
    
    print(f"Transformers version: {transformers.__version__}")
    print("=== Setup Complete ===")

if __name__ == "__main__":
    test_gpu_setup()                                                        