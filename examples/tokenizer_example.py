"""
Example usage of the BaseTokenizerWrapper module.

This script demonstrates how to use the tokenizer module for encoding,
decoding, and batch processing.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_tokenizers import BaseTokenizerWrapper


def main():
    # Initialize the tokenizer
    print("Initializing T5 tokenizer...")
    tokenizer = BaseTokenizerWrapper(model_name="t5-small")
    print(f"Tokenizer type: {type(tokenizer.tokenizer)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}\n")
    
    # Example 1: Basic encoding/decoding
    print("=" * 50)
    print("Example 1: Basic encoding/decoding")
    print("=" * 50)
    
    sentence = "hello, this is a sentence!"
    tokens = tokenizer.encode(sentence)
    print(f"Original text: {sentence}")
    print(f"Token IDs: {tokens['input_ids']}")
    print(f"Attention mask: {tokens['attention_mask']}")
    print(f"Decoded text: {tokenizer.decode(tokens['input_ids'])}")
    print(f"Tokenized: {tokenizer.tokenize(sentence)}\n")
    
    # Example 2: Special tokens
    print("=" * 50)
    print("Example 2: Special tokens")
    print("=" * 50)
    
    special_tokens = tokenizer.get_special_tokens()
    print("Special tokens:")
    for key, value in special_tokens.items():
        print(f"  {key}: {value}")
    print()
    
    # Example 3: Batch encoding
    print("=" * 50)
    print("Example 3: Batch encoding")
    print("=" * 50)
    
    sentences = [
        "this is the first sentence",
        "instead, this is the second sequence!"
    ]
    
    # Without padding
    tokens_no_padding = tokenizer.encode(sentences, padding=False)
    print("Without padding:")
    for i, (tok, att) in enumerate(zip(tokens_no_padding["input_ids"], 
                                       tokens_no_padding["attention_mask"])):
        print(f"  Sentence {i+1}: {tok} (len={len(tok)})")
    
    # With padding
    tokens_with_padding = tokenizer.encode(sentences, padding=True, return_tensors="pt")
    print("\nWith padding (PyTorch tensors):")
    print(f"  Input IDs shape: {tokens_with_padding['input_ids'].shape}")
    print(f"  Input IDs:\n{tokens_with_padding['input_ids']}")
    print(f"  Attention mask:\n{tokens_with_padding['attention_mask']}")
    
    # Batch decode
    decoded = tokenizer.batch_decode(tokens_with_padding['input_ids'])
    print("\nBatch decoded:")
    for i, text in enumerate(decoded):
        print(f"  Sentence {i+1}: {text}")
    print()
    
    # Example 4: Vocabulary access
    print("=" * 50)
    print("Example 4: Vocabulary access")
    print("=" * 50)
    
    vocab = tokenizer.vocabulary
    reverse_vocab = tokenizer.reverse_vocab
    
    # Check EOS token
    eos_token = "</s>"
    eos_id = vocab.get(eos_token)
    print(f"EOS token '{eos_token}' has ID: {eos_id}")
    print(f"Token ID {eos_id} maps to: '{reverse_vocab.get(eos_id)}'")
    print()
    
    # Example 5: Using tokenizer as callable
    print("=" * 50)
    print("Example 5: Using tokenizer as callable")
    print("=" * 50)
    
    result = tokenizer("translate english to german: hello, how are you?")
    print(f"Token IDs: {result['input_ids']}")
    print(f"Decoded: {tokenizer.decode(result['input_ids'])}")


if __name__ == "__main__":
    main()

