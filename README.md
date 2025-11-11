[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SabraHashemi/llm-project/blob/main/colab/colab_example.ipynb)

# LLM Base Project

A minimal, modular LLM project with:
- Tokenizer wrapper (`llm_tokenizers/BaseTokenizerWrapper`) using Hugging Face `AutoTokenizer`
- Model loader (`llm_models/BaseModelLoader`, `Seq2SeqModelLoader`) using `AutoModel*`
- Example scripts and tutorial-style tests
- One-click Colab notebook

## Quick Start (Colab)

Click the badge above or open `colab_example.ipynb`. It installs dependencies, loads tokenizer and model, and runs a small generation demo.

If you prefer running commands in a fresh Colab:

```python
!git clone https://github.com/SabraHashemi/llm-project.git
%cd llm-project
%pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil

import sys
sys.path.insert(0, '.')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader

tokenizer = BaseTokenizerWrapper("t5-small")
model = Seq2SeqModelLoader("t5-small")
print("✅ Ready")
```

## Quick Start (Local)

```powershell
# Create venv (Windows PowerShell)
python -m venv .venv
\.venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt

# Run tutorial-style tests
python test_t5_notebook.py

# Or smaller tests
python test_tokenizer.py
python test_model_loader.py
```

## Project Structure

```
llm-project/
  llm_tokenizers/
    __init__.py
    base_tokenizer.py         # BaseTokenizerWrapper (AutoTokenizer)
  llm_models/
    __init__.py
    base_model_loader.py      # BaseModelLoader, Seq2SeqModelLoader
  examples/
    tokenizer_example.py
    tokenizer_comparison.py
    model_loader_example.py
    token_embeddings_visualization.py   # PCA + cosine similarity
  labs/
    solution-01-t5.ipynb      # Reference notebook for T5
    solution-02-attention.ipynb
  tests & scripts
    test_tokenizer.py
    test_model_loader.py
    test_t5_notebook.py       # Comprehensive tutorial-style test
    train.py                  # Placeholder training entrypoint
    eval.py                   # Placeholder evaluation entrypoint
  README.md
  requirements.txt
```

## What’s Included

- Tokenization utilities with special tokens access, batch encode/decode, PyTorch tensors support
- Model loading for common tasks: seq2seq (T5/BART), causal (GPT-2), base encoder models (BERT)
- Examples and visualizations for embeddings (PCA, cosine similarity)
- Tests mirroring the reference T5 notebook (forward pass, generation, configuration)

## Examples

Run examples locally or in Colab:

```bash
python examples/tokenizer_example.py
python examples/model_loader_example.py
python examples/tokenizer_comparison.py
python examples/token_embeddings_visualization.py
```

## Notes

- This project is LLM-focused. Vision dataset helpers (`data/loader.py`, `data/mnist.py`) were removed to keep the codebase clean.
- Models are downloaded automatically on first use from Hugging Face.

## Troubleshooting

- ModuleNotFoundError: ensure your path is set (e.g., `sys.path.insert(0, '.')` in Colab or activate your venv locally).
- Slow first run: model/tokenizer downloads; subsequent runs are faster.

