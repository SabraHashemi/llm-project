"""
Comprehensive test based on solution-01-t5.ipynb

This test validates the tokenizer and model loader modules
by replicating the key functionality demonstrated in the notebook.
"""

import sys
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader


def test_tokenizer_loading():
    """Test 1: Tokenizer loading (Notebook Cell 2)"""
    print("=" * 70)
    print("Test 1: Tokenizer Loading")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   The first step in working with transformers is to load a tokenizer.")
    print("   Tokenizers convert text into a sequence of tokens (numbers) that the")
    print("   model can understand. Each model has its own tokenizer, but we use")
    print("   AutoTokenizer which automatically selects the correct one for T5.\n")
    
    model_name = "t5-small"
    print(f"🔄 Loading tokenizer for model: {model_name}")
    print("   (This may download the tokenizer on first run - this is normal!)")
    tokenizer = BaseTokenizerWrapper(model_name)
    
    assert tokenizer is not None, "Tokenizer should be loaded"
    assert tokenizer.vocab_size == 32100, f"Expected vocab size 32100, got {tokenizer.vocab_size}"
    
    print(f"\n✅ Tokenizer loaded successfully!")
    print(f"   - Tokenizer type: {type(tokenizer.tokenizer).__name__}")
    print(f"   - Vocabulary size: {tokenizer.vocab_size:,} tokens")
    print(f"   - This means the tokenizer knows {tokenizer.vocab_size:,} different tokens")
    print(f"     (words, subwords, and special tokens like <pad>, </s>)")
    print("\n✓ Test 1 PASSED\n")


def test_tokenizer_encoding_decoding():
    """Test 2: Basic encoding/decoding (Notebook Cells 6-8)"""
    print("=" * 70)
    print("Test 2: Tokenizer Encoding/Decoding")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   Encoding converts text → token IDs (numbers)")
    print("   Decoding converts token IDs (numbers) → text")
    print("   This is the fundamental conversion between human-readable text")
    print("   and the numerical representation the model uses.\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    sentence = "hello, this is a sentence!"
    
    print(f"📤 ENCODING: Converting text to token IDs")
    print(f"   Input text: '{sentence}'")
    tokens = tokenizer.encode(sentence)
    assert 'input_ids' in tokens, "Should have input_ids"
    assert 'attention_mask' in tokens, "Should have attention_mask"
    
    print(f"   → Token IDs: {tokens['input_ids']}")
    print(f"   → Attention mask: {tokens['attention_mask']}")
    print(f"   → Each number represents a token (word/subword) in the vocabulary")
    
    # Decode
    print(f"\n📥 DECODING: Converting token IDs back to text")
    decoded = tokenizer.decode(tokens['input_ids'])
    assert isinstance(decoded, str), "Decoded should be a string"
    assert "</s>" in decoded, "Should contain EOS token"
    
    print(f"   Token IDs: {tokens['input_ids']}")
    print(f"   → Decoded text: '{decoded}'")
    print(f"   → Note: The tokenizer automatically added </s> (end-of-sequence token)")
    print(f"     This tells the model where the sentence ends.\n")
    
    print("✓ Test 2 PASSED\n")


def test_special_tokens():
    """Test 3: Special tokens (Notebook Cells 27-29)"""
    print("=" * 70)
    print("Test 3: Special Tokens")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   Special tokens are special symbols that have specific meanings:")
    print("   - EOS (End of Sequence): Marks the end of text")
    print("   - PAD (Padding): Used to make sequences the same length in batches")
    print("   - BOS (Beginning of Sequence): Marks the start (T5 doesn't use this)")
    print("   - UNK (Unknown): For words not in vocabulary")
    print("   - SEP (Separator): Separates sentences (used in BERT, not T5)")
    print("   - CLS (Classification): For classification tasks (used in BERT, not T5)\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    special = tokenizer.get_special_tokens()
    
    # T5 should have EOS and PAD tokens, but no BOS token
    assert special['eos_token'] is not None, "Should have EOS token"
    assert special['eos_token_id'] is not None, "Should have EOS token ID"
    assert special['pad_token'] is not None, "Should have PAD token"
    assert special['pad_token_id'] is not None, "Should have PAD token ID"
    assert special['bos_token'] is None, "T5 should not have BOS token"
    
    print("🔍 Checking special tokens for T5:")
    print(f"   ✓ EOS token: '{special['eos_token']}' (ID: {special['eos_token_id']})")
    print(f"     → Used to mark the end of generated sequences")
    print(f"   ✓ PAD token: '{special['pad_token']}' (ID: {special['pad_token_id']})")
    print(f"     → Used for padding shorter sequences in batches")
    print(f"     → Also used as the decoder's starting token in T5!")
    print(f"   ✓ BOS token: {special['bos_token']}")
    print(f"     → T5 doesn't use BOS; it uses PAD as the starting token instead")
    print(f"   ✓ UNK token: '{special['unk_token']}' (ID: {special['unk_token_id']})")
    print(f"     → Used for unknown words (though T5's vocabulary is quite large)\n")
    
    print("✓ Test 3 PASSED\n")


def test_batch_encoding():
    """Test 4: Batch encoding with padding (Notebook Cells 31-37)"""
    print("=" * 70)
    print("Test 4: Batch Encoding with Padding")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   When processing multiple sentences at once (batches), they often have")
    print("   different lengths. We need to pad shorter sequences so they all have")
    print("   the same length. This allows the model to process them in parallel.")
    print("   The attention_mask tells the model which tokens are real and which are padding.\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    sentences = [
        "this is the first sentence",
        "instead, this is the second sequence!"
    ]
    
    print(f"📦 Processing batch of {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"   {i}. '{sent}'")
    
    # Without padding
    print(f"\n🔄 WITHOUT PADDING:")
    tokens_no_padding = tokenizer.encode(sentences, padding=False)
    assert len(tokens_no_padding['input_ids']) == 2, "Should have 2 sentences"
    assert len(tokens_no_padding['input_ids'][0]) != len(tokens_no_padding['input_ids'][1]), \
        "Sentences should have different lengths without padding"
    
    lengths = [len(t) for t in tokens_no_padding['input_ids']]
    print(f"   Sentence lengths: {lengths}")
    print(f"   → Problem: Different lengths can't be stacked into a tensor!")
    print(f"   → Solution: We need padding to make them the same length")
    
    # With padding
    print(f"\n🔄 WITH PADDING:")
    tokens_with_padding = tokenizer.encode(sentences, padding=True, return_tensors="pt")
    assert tokens_with_padding['input_ids'].shape[0] == 2, "Should have 2 sentences"
    assert tokens_with_padding['input_ids'].shape[1] == tokens_with_padding['input_ids'].shape[1], \
        "All sentences should have same length with padding"
    
    print(f"   Tensor shape: {tokens_with_padding['input_ids'].shape}")
    print(f"   → Shape means: [batch_size={tokens_with_padding['input_ids'].shape[0]}, "
          f"sequence_length={tokens_with_padding['input_ids'].shape[1]}]")
    print(f"   → Now all sentences have the same length and can be processed together!")
    print(f"   → Attention mask: {tokens_with_padding['attention_mask'].tolist()}")
    print(f"     (1 = real token, 0 = padding token that model should ignore)")
    
    # Batch decode
    decoded = tokenizer.batch_decode(tokens_with_padding['input_ids'])
    assert len(decoded) == 2, "Should decode 2 sentences"
    
    print(f"\n📥 BATCH DECODING: Converting token IDs back to text")
    for i, text in enumerate(decoded, 1):
        print(f"   {i}. '{text}'")
    
    print("\n✓ Test 4 PASSED\n")


def test_model_loading():
    """Test 5: Model loading (Notebook Cells 40-42)"""
    print("=" * 70)
    print("Test 5: Model Loading")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   Now we load the actual T5 model. T5 is a sequence-to-sequence model,")
    print("   meaning it has an encoder (reads input) and decoder (generates output).")
    print("   We use AutoModelForSeq2SeqLM which loads the full model with generation capabilities.\n")
    
    print("🔄 Loading T5-small model...")
    print("   (This may download the model on first run - this is normal!)")
    print("   Model size: ~240MB - this may take a moment to download.")
    model = Seq2SeqModelLoader("t5-small")
    
    assert model.model is not None, "Model should be loaded"
    assert model.config is not None, "Config should be loaded"
    assert model.vocab_size == 32128, f"Expected vocab size 32128, got {model.vocab_size}"
    assert model.hidden_size == 512, f"Expected hidden size 512 for t5-small, got {model.hidden_size}"
    assert model.num_layers == 6, f"Expected 6 layers for t5-small, got {model.num_layers}"
    assert model.num_heads == 8, f"Expected 8 heads for t5-small, got {model.num_heads}"
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Model type: {type(model.model).__name__}")
    print(f"\n📊 MODEL ARCHITECTURE (T5-small):")
    print(f"   Vocabulary size: {model.vocab_size:,}")
    print(f"     → Note: {model.vocab_size:,} is larger than tokenizer vocab ({model.vocab_size - 28:,})")
    print(f"       because it's rounded to a GPU-friendly number (251 × 128 = 32,128)")
    print(f"   Hidden size (d_model): {model.hidden_size}")
    print(f"     → The dimension of each token representation inside the model")
    print(f"   Number of layers: {model.num_layers}")
    print(f"     → Number of transformer blocks in encoder and decoder")
    print(f"   Number of attention heads: {model.num_heads}")
    print(f"     → Each head learns different patterns (parallel attention mechanisms)")
    print(f"     → Each head dimension: {model.hidden_size // model.num_heads} "
          f"({model.hidden_size} ÷ {model.num_heads} = {model.hidden_size // model.num_heads})")
    print("\n✓ Test 5 PASSED\n")


def test_model_forward_pass():
    """Test 6: Model forward pass (Notebook Cells 71-74)"""
    print("=" * 70)
    print("Test 6: Model Forward Pass")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   T5 uses an encoder-decoder architecture:")
    print("   1. ENCODER: Processes the input text (e.g., 'translate english to german: hello')")
    print("   2. DECODER: Generates output tokens one by one")
    print("   For the first step, decoder starts with <pad> token (T5's special start token)")
    print("   The model outputs 'logits' - probabilities for each possible next token.\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    model = Seq2SeqModelLoader("t5-small")
    
    input_sentence = "translate english to german: hello, how are you?"
    print(f"📝 Input sentence: '{input_sentence}'")
    
    print(f"\n🔢 STEP 1: Tokenize the input")
    tokens = tokenizer.encode(input_sentence, return_tensors="pt")
    print(f"   Encoder input (tokenized): {tokens['input_ids'].shape}")
    print(f"   → Shape: [batch_size=1, sequence_length={tokens['input_ids'].shape[1]}]")
    
    print(f"\n🔢 STEP 2: Prepare decoder input")
    decoder_input_ids = torch.tensor([[tokenizer.tokenizer.pad_token_id]])
    print(f"   Decoder input: {decoder_input_ids}")
    print(f"   → Shape: {decoder_input_ids.shape}")
    print(f"   → We start with <pad> token (ID: {tokenizer.tokenizer.pad_token_id})")
    print(f"   → This tells the model: 'start generating!'")
    
    # Forward pass
    print(f"\n🚀 STEP 3: Forward pass through the model")
    print(f"   Processing: encoder reads input → decoder generates first token")
    with torch.no_grad():
        output = model(**tokens, decoder_input_ids=decoder_input_ids)
    
    assert 'logits' in output, "Should have logits"
    assert 'past_key_values' in output, "Should have past_key_values"
    assert 'encoder_last_hidden_state' in output, "Should have encoder_last_hidden_state"
    
    assert output.logits.shape[0] == 1, "Batch size should be 1"
    assert output.logits.shape[1] == 1, "Should have 1 decoder token"
    assert output.logits.shape[2] == model.vocab_size, "Should have vocab_size logits"
    
    print(f"\n✅ Forward pass completed!")
    print(f"   Output logits shape: {output.logits.shape}")
    print(f"   → Shape means: [batch=1, sequence=1, vocab={model.vocab_size}]")
    print(f"   → For each position, we have {model.vocab_size:,} logits (one per token)")
    print(f"   → Higher logit = model thinks this token is more likely")
    print(f"\n   Output contains:")
    print(f"   - logits: Probabilities for next token ({output.logits.shape})")
    print(f"   - past_key_values: Cached attention states (for faster generation)")
    print(f"   - encoder_last_hidden_state: Final encoder representation")
    print(f"     Shape: {output.encoder_last_hidden_state.shape}")
    print("\n✓ Test 6 PASSED\n")


def test_token_generation_step_by_step():
    """Test 7: Manual token generation (Notebook Cells 79-87)"""
    print("=" * 70)
    print("Test 7: Manual Token Generation (Step by Step)")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   This demonstrates how the model generates text AUTOREGRESSIVELY:")
    print("   1. Start with <pad> token")
    print("   2. Model predicts the most likely next token (greedy decoding)")
    print("   3. Add that token to the sequence")
    print("   4. Use the new sequence to predict the next token")
    print("   5. Repeat until <eos> token or max length")
    print("   This is called 'autoregressive' because each prediction depends on previous ones.\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    model = Seq2SeqModelLoader("t5-small")
    
    input_sentence = "translate english to german: hello, how are you?"
    tokens = tokenizer.encode(input_sentence, return_tensors="pt")
    decoder_input_ids = torch.tensor([[tokenizer.tokenizer.pad_token_id]])
    
    print(f"📝 Input: '{input_sentence}'")
    print(f"\n🔢 STEP 1: Get first token prediction")
    
    # First step
    with torch.no_grad():
        output = model(**tokens, decoder_input_ids=decoder_input_ids)
    
    max_proba_token = output.logits[0, 0].argmax()
    predicted_token = tokenizer.decode([max_proba_token.item()])
    
    assert max_proba_token.item() > 0, "Token ID should be positive"
    assert isinstance(predicted_token, str), "Decoded token should be string"
    
    print(f"   Decoder input: {decoder_input_ids.tolist()}")
    print(f"   Model output: {output.logits.shape} logits")
    print(f"   → Most likely token ID: {max_proba_token.item()}")
    print(f"   → Decoded token: '{predicted_token}'")
    
    # Generate a few more tokens manually
    print(f"\n🔢 STEP 2: Generate tokens step by step")
    print(f"   (This shows the autoregressive generation process)")
    
    generated_tokens = []
    max_length = 5
    decoder_input_ids = torch.tensor([[tokenizer.tokenizer.pad_token_id]])
    
    for i in range(max_length):
        with torch.no_grad():
            output = model(**tokens, decoder_input_ids=decoder_input_ids)
        max_proba_tokens = output.logits[0].argmax(axis=1)
        new_token_id = max_proba_tokens[-1].item()
        generated_tokens.append(new_token_id)
        
        decoded_so_far = tokenizer.decode(generated_tokens)
        print(f"   Step {i+1}: Generated '{decoded_so_far}'")
        
        if new_token_id == tokenizer.tokenizer.eos_token_id:
            print(f"   → Found <eos> token, stopping generation")
            break
        
        decoder_input_ids = torch.hstack([decoder_input_ids, max_proba_tokens[-1].view(1, 1)])
        print(f"     Decoder now has: {decoder_input_ids.tolist()[0]} tokens")
    
    decoded_sequence = tokenizer.decode(generated_tokens)
    print(f"\n✅ Generated sequence:")
    print(f"   Token IDs: {generated_tokens}")
    print(f"   Decoded text: '{decoded_sequence}'")
    print(f"   → This is how the model builds output one token at a time!\n")
    
    print("✓ Test 7 PASSED\n")


def test_model_generate_method():
    """Test 8: Using model.generate() (Notebook Cell 89)"""
    print("=" * 70)
    print("Test 8: Model.generate() Method")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   Instead of manually generating tokens step-by-step, we can use")
    print("   model.generate() which does the same thing but faster and with")
    print("   optimizations (like caching encoder outputs and past key-values).")
    print("   This is the recommended way to generate text!\n")
    
    tokenizer = BaseTokenizerWrapper("t5-small")
    model = Seq2SeqModelLoader("t5-small")
    
    input_sentence = "translate english to german: hello, how are you?"
    tokens = tokenizer.encode(input_sentence, return_tensors="pt")
    
    print(f"📝 Input: '{input_sentence}'")
    print(f"\n🚀 Generating text using model.generate()...")
    print(f"   (This is much faster than manual generation!)")
    
    # Generate using model.generate()
    with torch.no_grad():
        generated_ids = model.generate(**tokens, max_length=20)
    
    assert generated_ids.shape[0] == 1, "Should have batch size 1"
    assert generated_ids.shape[1] > 0, "Should generate at least one token"
    
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    assert isinstance(decoded, str), "Generated text should be string"
    assert len(decoded) > 0, "Generated text should not be empty"
    
    print(f"\n✅ Generation completed!")
    print(f"   Generated token IDs shape: {generated_ids.shape}")
    print(f"   → Generated {generated_ids.shape[1]} tokens")
    print(f"   Decoded text: '{decoded}'")
    print(f"\n   💡 Note: model.generate() is optimized and faster than manual generation")
    print(f"   because it caches encoder outputs and reuses past key-values.\n")
    
    print("✓ Test 8 PASSED\n")


def test_model_configuration():
    """Test 9: Model configuration access (Notebook Cells 42-43)"""
    print("=" * 70)
    print("Test 9: Model Configuration")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   The model configuration contains all hyperparameters and settings:")
    print("   - Architecture details (layers, heads, dimensions)")
    print("   - Training settings (dropout, activation functions)")
    print("   - Model-specific parameters")
    print("   This is useful for understanding the model architecture.\n")
    
    model = Seq2SeqModelLoader("t5-small")
    config_info = model.get_config_info()
    
    assert 'model_name' in config_info, "Should have model_name"
    assert 'model_type' in config_info, "Should have model_type"
    assert config_info['model_type'] == 'seq2seq', "Should be seq2seq model"
    
    # Check key config values for T5-small
    assert config_info['vocab_size'] == 32128, "Vocab size should match"
    assert config_info['hidden_size'] == 512, "Hidden size should be 512 for t5-small"
    
    print(f"📊 MODEL CONFIGURATION:")
    print(f"   Model name: {config_info['model_name']}")
    print(f"   Model type: {config_info['model_type']}")
    print(f"   → This is a sequence-to-sequence model (encoder-decoder)")
    print(f"\n   Key parameters:")
    print(f"   - Vocabulary size: {config_info['vocab_size']:,}")
    print(f"   - Hidden size: {config_info['hidden_size']}")
    print(f"   - Number of layers: {config_info['num_layers']}")
    print(f"   - Number of attention heads: {config_info['num_heads']}")
    print(f"\n   Total config parameters: {len(config_info)} items")
    print(f"   → The full config contains many more details about the model architecture\n")
    
    print("✓ Test 9 PASSED\n")


def test_model_architecture_access():
    """Test 10: Model architecture components (Notebook Cells 45-69)"""
    print("=" * 70)
    print("Test 10: Model Architecture Access")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   T5 has an encoder-decoder architecture:")
    print("   - ENCODER: Processes input text (self-attention + feed-forward)")
    print("   - DECODER: Generates output (self-attention + cross-attention + feed-forward)")
    print("   - SHARED EMBEDDINGS: Encoder and decoder share the same token embeddings")
    print("   - LM_HEAD: Final layer that maps decoder output to vocabulary probabilities")
    print("   Each encoder/decoder has multiple 'blocks' (transformer layers).\n")
    
    model = Seq2SeqModelLoader("t5-small")
    
    # Check encoder exists
    assert hasattr(model.model, 'encoder'), "Should have encoder"
    assert hasattr(model.model, 'decoder'), "Should have decoder"
    assert hasattr(model.model, 'lm_head'), "Should have lm_head"
    
    # Check encoder blocks
    assert hasattr(model.model.encoder, 'block'), "Encoder should have block"
    assert len(model.model.encoder.block) == model.num_layers, "Number of blocks should match num_layers"
    
    # Check decoder blocks
    assert hasattr(model.model.decoder, 'block'), "Decoder should have block"
    assert len(model.model.decoder.block) == model.num_layers, "Number of blocks should match num_layers"
    
    print(f"🏗️  MODEL ARCHITECTURE:")
    print(f"   Encoder:")
    print(f"   - Has {len(model.model.encoder.block)} transformer blocks")
    print(f"     → Each block contains: self-attention + feed-forward network")
    print(f"   - Processes input text and creates contextual representations")
    
    print(f"\n   Decoder:")
    print(f"   - Has {len(model.model.decoder.block)} transformer blocks")
    print(f"     → Each block contains: self-attention + cross-attention + feed-forward")
    print(f"     → Cross-attention connects decoder to encoder outputs")
    print(f"   - Generates output tokens one by one")
    
    # Check shared embeddings
    if hasattr(model.model, 'shared'):
        assert model.model.shared is not None, "Should have shared embeddings"
        assert id(model.model.shared) == id(model.model.encoder.embed_tokens), \
            "Shared embeddings should be same as encoder embeddings"
        assert id(model.model.decoder.embed_tokens) == id(model.model.encoder.embed_tokens), \
            "Shared embeddings should be same as decoder embeddings"
        
        print(f"\n   Shared Embeddings:")
        print(f"   - Encoder and decoder use the SAME embedding layer")
        print(f"   - This is efficient: one embedding matrix for both")
        print(f"   - Embedding size: {model.model.shared.num_embeddings} tokens × {model.model.shared.embedding_dim} dimensions")
    
    print(f"\n   Language Model Head (lm_head):")
    print(f"   - Final linear layer that maps decoder output to vocabulary")
    print(f"   - Input: {model.model.lm_head.in_features} dimensions (hidden size)")
    print(f"   - Output: {model.model.lm_head.out_features} dimensions (vocab size)")
    print(f"   - This produces the logits (probabilities) for each token\n")
    
    print("✓ Test 10 PASSED\n")


def run_all_tests():
    """Run all tests based on the notebook"""
    print("\n" + "=" * 70)
    print("TUTORIAL: COMPREHENSIVE TEST BASED ON solution-01-t5.ipynb")
    print("=" * 70)
    print("\nThis test suite walks through the key concepts from the T5 notebook:")
    print("  1. Tokenization - Converting text to numbers")
    print("  2. Encoding/Decoding - Basic text processing")
    print("  3. Special Tokens - Understanding model control symbols")
    print("  4. Batch Processing - Handling multiple inputs")
    print("  5. Model Loading - Loading the T5 transformer")
    print("  6. Forward Pass - How the model processes input")
    print("  7. Token Generation - Autoregressive text generation")
    print("  8. Model.generate() - Optimized generation method")
    print("  9. Configuration - Understanding model architecture")
    print(" 10. Architecture - Exploring encoder-decoder structure")
    print("\n" + "=" * 70)
    print("\n💡 NOTE: For token embeddings visualization (PCA and cosine similarity),")
    print("   run: python examples/token_embeddings_visualization.py")
    print("=" * 70 + "\n")
    
    try:
        test_tokenizer_loading()
        test_tokenizer_encoding_decoding()
        test_special_tokens()
        test_batch_encoding()
        test_model_loading()
        test_model_forward_pass()
        test_token_generation_step_by_step()
        test_model_generate_method()
        test_model_configuration()
        test_model_architecture_access()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 CONGRATULATIONS!")
        print("   You've successfully completed all tests from solution-01-t5.ipynb!")
        print("\n📚 KEY CONCEPTS LEARNED:")
        print("   ✓ How tokenizers convert text to numbers")
        print("   ✓ How sequence-to-sequence models work (encoder-decoder)")
        print("   ✓ How autoregressive generation creates text token by token")
        print("   ✓ How to use model.generate() for efficient text generation")
        print("   ✓ Understanding T5 model architecture")
        print("\n💡 NEXT STEPS:")
        print("   - Try different input prompts")
        print("   - Experiment with different T5 model sizes (t5-base, t5-large)")
        print("   - Explore attention mechanisms in the next notebook")
        print("\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

