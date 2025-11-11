# How to Run Your Project in Colab

## Step-by-Step Guide

### Step 1: Install Dependencies
```python
!pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil
```

### Step 2: Setup Python Path
```python
import sys
sys.path.insert(0, '/content')   # Or '.' if you cloned into /content/llm-project
```

### Step 3: Import Your Modules
```python
from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

### Step 4: Run Examples/Tests
```python
!python test_t5_notebook.py
!python examples/tokenizer_example.py
```


