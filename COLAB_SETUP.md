# Running This Project in Google Colab

This guide shows you how to use your modular LLM project in Google Colab.

## Method 1: Clone from GitHub (Recommended)

### Step 1: Upload to GitHub (if not already)

1. Create a GitHub repository
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Use in Colab

```python
# In a Colab cell
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Install dependencies
!pip install -q transformers torch matplotlib scikit-learn numpy

# Import your modules
import sys
sys.path.insert(0, '.')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

## Method 2: Upload Files Directly

### Step 1: Upload to Colab

1. In Colab, click the folder icon (📁) on the left sidebar
2. Click "Upload" and select these folders/files:
   - `llm_tokenizers/` folder
   - `llm_models/` folder
   - Any example files you want to use

### Step 2: Setup in Colab

```python
# Install dependencies
!pip install -q transformers torch matplotlib scikit-learn numpy

# Add to Python path
import sys
sys.path.insert(0, '/content')

# Import your modules
from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

## Method 3: Use Google Drive

### Step 1: Upload to Google Drive

1. Upload your project folder to Google Drive
2. Name it something like `llm_project`

### Step 2: Mount Drive in Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
import os
os.chdir('/content/drive/MyDrive/llm_project')

# Install dependencies
!pip install -q transformers torch matplotlib scikit-learn numpy

# Import modules
import sys
sys.path.insert(0, '/content/drive/MyDrive/llm_project')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

## Quick Start Example

Once set up, you can use your modules like this:

```python
# Initialize tokenizer
tokenizer = BaseTokenizerWrapper("t5-small")

# Initialize model
model = Seq2SeqModelLoader("t5-small")

# Generate text
import torch
input_text = "translate english to german: hello"
tokens = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(**tokens, max_length=20)

result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Input: {input_text}")
print(f"Output: {result}")
```

## Using the Example Notebook

1. Open `colab_example.ipynb` in Colab
2. Follow the cells step by step
3. Adjust paths as needed for your setup

## Tips

1. **GPU Acceleration**: Colab provides free GPU! Enable it:
   - Runtime → Change runtime type → GPU

2. **Save Your Work**: 
   - Save notebooks to Drive
   - Or download `.ipynb` files

3. **Large Models**: 
   - Models download automatically on first use
   - They're cached in `/root/.cache/huggingface/`

4. **File Persistence**: 
   - Colab sessions reset when disconnected
   - Use Google Drive for persistent storage

## Troubleshooting

### Import Error
```python
# Make sure path is correct
import sys
print(sys.path)  # Check current paths
sys.path.insert(0, '/content/YOUR_PROJECT_PATH')
```

### Module Not Found
```python
# Install missing packages
!pip install PACKAGE_NAME
```

### Path Issues
```python
# Check current directory
import os
print(os.getcwd())

# Change directory if needed
os.chdir('/content/YOUR_PATH')
```

## Example: Full Workflow

```python
# 1. Setup
!pip install -q transformers torch matplotlib scikit-learn numpy
import sys
sys.path.insert(0, '/content')

# 2. Import
from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader

# 3. Use
tokenizer = BaseTokenizerWrapper("t5-small")
model = Seq2SeqModelLoader("t5-small")

# 4. Generate
import torch
text = "translate english to german: hello world"
tokens = tokenizer.encode(text, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**tokens, max_length=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Enjoy using your modular LLM project in Colab! 🚀

