# Summary: solution-01-t5.ipynb

## Overview
This notebook is a comprehensive tutorial on working with T5 (Text-To-Text Transfer Transformer), a sequence-to-sequence model. It teaches you how to use tokenizers, load models, understand their architecture, and generate text.

---

## Main Sections

### 1. **Tokenization** (Cells 0-37)
**What it teaches:**
- How to load a tokenizer using `AutoTokenizer`
- Converting text → token IDs (encoding)
- Converting token IDs → text (decoding)
- Understanding vocabulary (32,100 tokens for T5)
- Special tokens: `<pad>`, `</s>` (EOS), etc.
- Batch processing with padding
- Attention masks (to ignore padding tokens)

**Key concepts:**
- T5 uses SentencePiece tokenization (see the `▁` prefix indicating word starts)
- Vocabulary size: 32,000 base tokens + 100 special tokens (`<extra_id_0>` to `<extra_id_99>`)
- EOS token (`</s>`) is automatically added when encoding
- Padding allows processing multiple sentences of different lengths together

---

### 2. **Model Analysis** (Cells 38-69)
**What it teaches:**
- Loading T5 model with `AutoModelForSeq2SeqLM`
- Understanding model configuration (hidden size, layers, heads)
- Exploring model architecture:
  - **Encoder**: Processes input (self-attention + feed-forward)
  - **Decoder**: Generates output (self-attention + cross-attention + feed-forward)
  - **Shared embeddings**: Encoder and decoder share the same token embeddings
  - **lm_head**: Final layer that maps decoder output to vocabulary probabilities

**Key architecture details (T5-small):**
- Hidden size (d_model): 512
- Number of layers: 6 (encoder + decoder)
- Attention heads: 8
- Feed-forward dimension (d_ff): 2,048
- Vocabulary size: 32,128 (rounded for GPU efficiency)

---

### 3. **Token Embeddings Visualization** (Cells 47-55)
**What it teaches:**
- Extracting token embeddings from `model.shared`
- Using PCA to visualize embeddings in 2D
- Computing cosine similarity between word embeddings
- Understanding semantic relationships (similar words cluster together)

**Example words used:**
- Furniture: chair, table, plate, knife, spoon
- Animals: horse, goat, sheep, cat, dog

**Insights:**
- Words with similar meanings have similar embeddings
- PCA shows semantic clustering (furniture vs animals)
- Cosine similarity quantifies how "related" words are

---

### 4. **Token Generation** (Cells 70-90)
**What it teaches:**
- How encoder-decoder models generate text
- Manual step-by-step generation (autoregressive)
- Using `model.generate()` for optimized generation
- Understanding decoder input (starts with `<pad>` token for T5)

**Process:**
1. Encoder processes input: "translate english to german: hello"
2. Decoder starts with `<pad>` token
3. Model predicts next token (greedy: highest probability)
4. Add predicted token to decoder input
5. Repeat until `<eos>` token or max length

**Example output:**
- Input: "translate english to german: hello, how are you?"
- Output: "Hallo, wie sind Sie?" (German translation)

---

### 5. **Cross-Attention Analysis** (Cells 91-107)
**What it teaches:**
- How decoder "attends" to encoder outputs
- Visualizing attention weights across layers
- Understanding what the model focuses on when generating

**Key insights:**
- Early layers focus on task identification ("translate", "german")
- Later layers focus on content words ("hello")
- Cross-attention connects decoder to encoder representations

---

## Key Learning Points

### 1. **AutoTokenizer & AutoModel**
- `AutoTokenizer`: Automatically loads the correct tokenizer for any model
- `AutoModelForSeq2SeqLM`: Loads sequence-to-sequence models (T5, BART, etc.)
- These are "smart" classes that detect model type automatically

### 2. **Encoder-Decoder Architecture**
- **Encoder**: Reads and understands input (bidirectional attention)
- **Decoder**: Generates output one token at a time (autoregressive)
- **Cross-attention**: Decoder looks at encoder outputs to generate relevant text

### 3. **Autoregressive Generation**
- Each token depends on previous tokens
- Process: Start with `<pad>` → predict token 1 → predict token 2 → ... → `<eos>`
- `model.generate()` optimizes this with caching

### 4. **Special Tokens**
- `<pad>`: Padding (also used as decoder start token in T5)
- `</s>`: End of sequence
- `<extra_id_0>` to `<extra_id_99>`: Used for T5's pre-training tasks

### 5. **Embeddings**
- Each token is represented as a high-dimensional vector (512D for T5-small)
- Similar words have similar embeddings
- Embeddings capture semantic meaning

---

## Practical Takeaways

1. **Tokenization is the foundation**: Text must be converted to numbers before the model can process it
2. **Batch processing requires padding**: Different length sentences need padding to form tensors
3. **Generation is iterative**: Models build output token by token, not all at once
4. **Attention shows what matters**: Visualizing attention reveals what the model focuses on
5. **Embeddings encode meaning**: Similar words cluster together in embedding space

---

## What Makes This Notebook Educational

- **Step-by-step**: Builds from basics (tokenization) to advanced (attention)
- **Hands-on**: Every concept is demonstrated with code
- **Visual**: Includes plots for embeddings and attention
- **Practical**: Shows real translation example
- **Deep dive**: Explores model internals (architecture, embeddings, attention)

This notebook is essentially a **complete course** on understanding and using T5 models!

