"""
Example showing how BaseTokenizerWrapper works with different models.

This demonstrates that BaseTokenizerWrapper works with ANY model:
- T5 models
- BERT models
- GPT-2 models
- And any other Hugging Face model!
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_tokenizers import BaseTokenizerWrapper


def main():
    print("=" * 70)
    print("BaseTokenizerWrapper - Works with ANY Model!")
    print("=" * 70)
    
    # ============================================================
    # Example 1: Using T5 model
    # ============================================================
    print("\n" + "=" * 70)
    print("Example 1: Using T5 model")
    print("=" * 70)
    
    t5_tokenizer = BaseTokenizerWrapper("t5-small")
    print(f"Model: t5-small")
    print(f"Type: {type(t5_tokenizer.tokenizer)}")
    print(f"Vocab size: {t5_tokenizer.vocab_size}")
    
    # Test encoding/decoding
    text = "translate english to german: hello"
    tokens = t5_tokenizer.encode(text)
    decoded = t5_tokenizer.decode(tokens['input_ids'])
    print(f"\nText: {text}")
    print(f"Tokens: {tokens['input_ids']}")
    print(f"Decoded: {decoded}")
    
    # ============================================================
    # Example 2: Using BERT model
    # ============================================================
    print("\n" + "=" * 70)
    print("Example 2: Using BERT model")
    print("=" * 70)
    
    bert_tokenizer = BaseTokenizerWrapper("bert-base-uncased")
    print(f"Model: bert-base-uncased")
    print(f"Type: {type(bert_tokenizer.tokenizer)}")
    print(f"Vocab size: {bert_tokenizer.vocab_size}")
    
    # Test encoding/decoding
    text = "hello world"
    tokens = bert_tokenizer.encode(text)
    decoded = bert_tokenizer.decode(tokens['input_ids'])
    print(f"\nText: {text}")
    print(f"Tokens: {tokens['input_ids']}")
    print(f"Decoded: {decoded}")
    
    # ============================================================
    # Example 3: Using GPT-2 model
    # ============================================================
    print("\n" + "=" * 70)
    print("Example 3: Using GPT-2 model")
    print("=" * 70)
    
    gpt2_tokenizer = BaseTokenizerWrapper("gpt2")
    print(f"Model: gpt2")
    print(f"Type: {type(gpt2_tokenizer.tokenizer)}")
    print(f"Vocab size: {gpt2_tokenizer.vocab_size}")
    
    # Test encoding/decoding
    text = "hello world"
    tokens = gpt2_tokenizer.encode(text)
    decoded = gpt2_tokenizer.decode(tokens['input_ids'])
    print(f"\nText: {text}")
    print(f"Tokens: {tokens['input_ids']}")
    print(f"Decoded: {decoded}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    BaseTokenizerWrapper:
    - ✅ Works with ANY model (BERT, GPT-2, T5, DistilBERT, etc.)
    - ✅ General purpose - one class for all models
    - ✅ AutoTokenizer automatically detects the correct tokenizer type
    - ✅ All methods work the same way regardless of model
    
    How it works:
    - AutoTokenizer.from_pretrained() automatically detects the model type
    - It returns the correct tokenizer class (T5TokenizerFast, BertTokenizerFast, etc.)
    - You just need to specify the model name!
    
    Usage:
    - For T5: BaseTokenizerWrapper("t5-small")
    - For BERT: BaseTokenizerWrapper("bert-base-uncased")
    - For GPT-2: BaseTokenizerWrapper("gpt2")
    - For any model: BaseTokenizerWrapper("model-name")
    """)


if __name__ == "__main__":
    main()
