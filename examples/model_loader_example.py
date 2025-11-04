"""
Example usage of the BaseModelLoader module.

This script demonstrates how to load and use transformer models.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_models import BaseModelLoader, Seq2SeqModelLoader
from llm_tokenizers import BaseTokenizerWrapper


def main():
    print("=" * 70)
    print("Model Loader Examples")
    print("=" * 70)
    
    # Example 1: Load T5 model (seq2seq)
    print("\n" + "=" * 70)
    print("Example 1: Loading T5 model (seq2seq)")
    print("=" * 70)
    
    model = Seq2SeqModelLoader("t5-small")
    print(f"Model type: {type(model.model)}")
    print(f"Model name: {model.model_name}")
    
    # Get config info
    config_info = model.get_config_info()
    print(f"\nConfiguration:")
    print(f"  Vocab size: {config_info['vocab_size']}")
    print(f"  Hidden size: {config_info['hidden_size']}")
    print(f"  Num layers: {config_info['num_layers']}")
    print(f"  Num heads: {config_info['num_heads']}")
    
    # Example 2: Load model with tokenizer and test forward pass
    print("\n" + "=" * 70)
    print("Example 2: Testing model with tokenizer")
    print("=" * 70)
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    
    # Prepare input
    input_text = "translate english to german: hello"
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    
    print(f"Input text: {input_text}")
    print(f"Token IDs: {tokens['input_ids']}")
    
    # Forward pass
    import torch
    decoder_input_ids = torch.tensor([[tokenizer.tokenizer.pad_token_id]])
    
    with torch.no_grad():
        output = model(**tokens, decoder_input_ids=decoder_input_ids)
    
    print(f"\nOutput logits shape: {output.logits.shape}")
    print(f"Output keys: {list(output.keys())}")
    
    # Get the predicted token
    predicted_token_id = output.logits[0, 0].argmax().item()
    predicted_token = tokenizer.decode([predicted_token_id])
    print(f"\nFirst predicted token ID: {predicted_token_id}")
    print(f"First predicted token: {predicted_token}")
    
    # Example 3: Using generate method
    print("\n" + "=" * 70)
    print("Example 3: Using model.generate() method")
    print("=" * 70)
    
    generated_ids = model.generate(**tokens, max_length=20)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")
    
    # Example 4: Load different model types
    print("\n" + "=" * 70)
    print("Example 4: Loading different model types")
    print("=" * 70)
    
    print("\n4a. Loading BERT base model:")
    bert_model = BaseModelLoader("bert-base-uncased", model_type="base")
    print(f"   Model type: {type(bert_model.model)}")
    print(f"   Hidden size: {bert_model.hidden_size}")
    
    print("\n4b. Loading GPT-2 (causal model):")
    gpt2_model = BaseModelLoader("gpt2", model_type="causal")
    print(f"   Model type: {type(gpt2_model.model)}")
    print(f"   Vocab size: {gpt2_model.vocab_size}")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

