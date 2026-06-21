import sys
import warnings

def verify_env():
    print("--- Environment Verification ---")
    
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA is available. Device 0: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available.")
    except ImportError:
        print("torch is not installed.")

    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except ImportError:
        print("transformers is not installed.")

    try:
        import peft
        print(f"peft version: {peft.__version__}")
    except ImportError:
        print("peft is not installed.")

    try:
        import datasets
        print(f"datasets version: {datasets.__version__}")
        
        # Check datasets version
        version_parts = datasets.__version__.split('.')
        if int(version_parts[0]) >= 4:
            print("WARNING: datasets version is >= 4.0. Script-based dataset loading is unavailable. CUAD must be loaded from raw JSON (expected for Phase 1).")
    except ImportError:
        print("datasets is not installed.")

    print("\n--- Testing Tokenizer Loads ---")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Cannot test tokenizers; transformers not installed.")
        return

    try:
        print("Loading sshleifer/distilbart-cnn-12-6 tokenizer...")
        tokenizer_db = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        print("Successfully loaded distilbart tokenizer.")
    except Exception as e:
        print(f"Failed to load distilbart tokenizer. Error: {e}")

    try:
        print("Loading t5-small tokenizer...")
        tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
        print("Successfully loaded t5-small tokenizer.")
    except Exception as e:
        print(f"Failed to load t5-small tokenizer. Error: {e}")

if __name__ == "__main__":
    verify_env()
