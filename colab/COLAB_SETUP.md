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

**⚠️ IMPORTANT: Make your repository PUBLIC first!**
1. Go to: https://github.com/SabraHashemi/llm-project/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Make public"

Then in Colab:

```python
# Clone repository (only works if public!)
!git clone https://github.com/SabraHashemi/llm-project.git
%cd llm-project

# Install dependencies
!pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil

# Import your modules
import sys
sys.path.insert(0, '.')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

**If repository is private**, see "Method 3: Upload Files Directly" below (no git needed!)

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


