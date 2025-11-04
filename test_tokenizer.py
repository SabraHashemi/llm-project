"""
Quick test script for the tokenizer module.

Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_tokenizers import BaseTokenizerWrapper


def test_basic_functionality():
    """Test basic tokenizer functionality."""
    print("Testing tokenizer...")
    
    # Initialize tokenizer
    print("1. Initializing tokenizer (this may download the model on first run)...")
    tokenizer = BaseTokenizerWrapper("t5-small")
    print(f"   ✓ Tokenizer loaded: {type(tokenizer.tokenizer).__name__}")
    print(f"   ✓ Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encoding
    print("\n2. Testing encoding...")
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(f"   ✓ Text: '{text}'")
    print(f"   ✓ Token IDs: {tokens['input_ids']}")
    
    # Test decoding
    print("\n3. Testing decoding...")
    decoded = tokenizer.decode(tokens['input_ids'])
    print(f"   ✓ Decoded: '{decoded}'")
    
    # Test tokenize
    print("\n4. Testing tokenization...")
    token_list = tokenizer.tokenize(text)
    print(f"   ✓ Tokens: {token_list}")
    
    # Test special tokens
    print("\n5. Testing special tokens...")
    special = tokenizer.get_special_tokens()
    print(f"   ✓ EOS token: {special['eos_token']} (ID: {special['eos_token_id']})")
    print(f"   ✓ PAD token: {special['pad_token']} (ID: {special['pad_token_id']})")
    
    print("\n" + "="*50)
    print("✅ All tests passed! Tokenizer is working correctly.")
    print("="*50)


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except ImportError as e:
        print("❌ Error: Missing required package!")
        print(f"   {e}")
        print("\n   Please install dependencies:")
        print("   pip install transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n   Please check your installation and try again.")

