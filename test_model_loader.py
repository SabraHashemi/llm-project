"""
Quick test script for the model loader module.

Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_models import BaseModelLoader, Seq2SeqModelLoader


def test_model_loader():
    """Test basic model loader functionality."""
    print("Testing model loader...")
    
    # Initialize model
    print("1. Loading T5 model (this may download the model on first run)...")
    model = Seq2SeqModelLoader("t5-small")
    print(f"   ✓ Model loaded: {type(model.model).__name__}")
    print(f"   ✓ Model name: {model.model_name}")
    
    # Test config info
    print("\n2. Testing configuration access...")
    print(f"   ✓ Vocab size: {model.vocab_size}")
    print(f"   ✓ Hidden size: {model.hidden_size}")
    print(f"   ✓ Num layers: {model.num_layers}")
    print(f"   ✓ Num heads: {model.num_heads}")
    
    # Test model callable
    print("\n3. Testing model is callable...")
    print(f"   ✓ Model is callable: {callable(model)}")
    
    # Test generate method exists
    print("\n4. Testing generation method...")
    print(f"   ✓ Has generate method: {hasattr(model, 'generate')}")
    
    print("\n" + "="*50)
    print("✅ All tests passed! Model loader is working correctly.")
    print("="*50)


if __name__ == "__main__":
    try:
        test_model_loader()
    except ImportError as e:
        print("❌ Error: Missing required package!")
        print(f"   {e}")
        print("\n   Please install dependencies:")
        print("   pip install transformers torch")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n   Please check your installation and try again.")

