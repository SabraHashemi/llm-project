# Working Locally, Running on Colab

This guide shows you how to develop your code locally but execute it on Colab for GPU access.

## 🎯 Best Workflow: GitHub Sync (Recommended)

### Setup
1. **Initialize Git** (if not already done):
```bash
git init
git add .
git commit -m "Initial commit"
```

2. **Create GitHub Repository**:
   - Go to GitHub and create a new repository
   - Don't initialize with README (you already have one)

3. **Connect and Push**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Daily Workflow

#### On Your Local Machine:
```bash
# Make changes to your code
# ... edit files ...

# Commit and push
git add .
git commit -m "Added new feature"
git push
```

#### In Colab:
```python
# Pull latest changes
!git pull

# Or clone fresh (if starting new session)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Install dependencies
%pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil

# Run your code
import sys
sys.path.insert(0, '.')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
# ... your code ...
```

## 📁 Alternative: Google Drive Sync

### Setup
1. **Install Google Drive Desktop**:
   - Download from: https://www.google.com/drive/download/
   - Install and sync a folder (e.g., `Google Drive/llm_project`)

2. **Work Locally**:
   - Edit files in your synced folder
   - Changes sync automatically to Drive

3. **Run in Colab**:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to synced folder
import os
os.chdir('/content/drive/MyDrive/llm_project')

# Install dependencies
%pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil

# Use your code
import sys
sys.path.insert(0, '/content/drive/MyDrive/llm_project')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

## 🔄 Hybrid Approach: Local Development + Colab Execution

### Option A: Quick Upload Script

Create a script to quickly upload your modules to Colab:

```python
# upload_to_colab.py
import shutil
import os

# Files to upload
files_to_upload = [
    'llm_tokenizers',
    'llm_models',
    'examples',
]

print("Ready to upload! Copy these files to Colab:")
for item in files_to_upload:
    if os.path.exists(item):
        print(f"  - {item}/")
```

### Option B: Manual Upload (Quick Testing)

1. **Develop locally** in your IDE
2. **Test in Colab**:
   - Upload only changed files via Colab's file browser
   - Or zip your modules and upload

```python
# In Colab, after uploading files:
import sys
sys.path.insert(0, '/content')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader
```

## 🛠️ VS Code Remote Development

### Using VS Code with Colab

1. **Install Jupyter Extension** in VS Code
2. **Connect to Colab**:
   - Open VS Code
   - Install "Colab" extension
   - Right-click on `.ipynb` file → "Open in Colab"

3. **Workflow**:
   - Edit `.py` files locally
   - Open notebook in Colab
   - Import your local modules (uploaded or synced)

## 📝 Recommended Workflow

### For Daily Development:

```
┌─────────────────┐
│  Local Machine  │
│  (Your IDE)     │
│                 │
│  - Write code   │
│  - Test logic   │
│  - Commit to Git│
└────────┬────────┘
         │
         │ git push
         ▼
┌─────────────────┐
│     GitHub      │
│   (Repository)  │
└────────┬────────┘
         │
         │ git clone/pull
         ▼
┌─────────────────┐
│  Google Colab   │
│  (GPU Runtime)  │
│                 │
│  - Pull code    │
│  - Run models   │
│  - Get results  │
└─────────────────┘
```

### Step-by-Step:

1. **Morning Setup** (Local):
   ```bash
   # Pull any changes from GitHub
   git pull
   ```

2. **Development** (Local):
   - Edit your code in VS Code/PyCharm/etc.
   - Test with small examples locally
   - Commit frequently

3. **Heavy Execution** (Colab):
   ```python
   # Pull latest code
   !git pull
   
   # Run your experiments
   !python test_t5_notebook.py
   # or
   !python examples/token_embeddings_visualization.py
   ```

4. **Evening Sync** (Local):
   ```bash
   # Pull any results/outputs from Colab
   git pull
   ```

## 💡 Pro Tips

### 1. Keep Colab Notebook for Execution Only
- Keep your main logic in `.py` files (local)
- Use Colab notebooks just for running experiments
- Import your modules, don't rewrite them

### 2. Use Environment Variables
```python
# In Colab, set environment variables
import os
os.environ['MODEL_NAME'] = 't5-small'
os.environ['MAX_LENGTH'] = '50'

# In your local code, use them
import os
model_name = os.getenv('MODEL_NAME', 't5-small')
```

### 3. Save Results to Drive
```python
# In Colab, save outputs to Drive
from google.colab import drive
drive.mount('/content/drive')

# Save plots, results, etc.
plt.savefig('/content/drive/MyDrive/results/plot.png')
```

### 4. Quick Sync Script
Create a helper script for frequent syncing:

```python
# sync_to_colab.py
import subprocess

def sync_to_colab():
    """Quick sync: commit and push to GitHub"""
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', 'Auto-sync to Colab'])
    subprocess.run(['git', 'push'])
    print("✅ Synced to GitHub! Now pull in Colab.")

if __name__ == "__main__":
    sync_to_colab()
```

## 🚀 Quick Start Template

### Colab Notebook Cell 1:
```python
# Setup
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
%pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil
```

### Colab Notebook Cell 2:
```python
# Import
import sys
sys.path.insert(0, '.')

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader

print("✅ Ready to use your modules!")
```

### Colab Notebook Cell 3:
```python
# Use your code
tokenizer = BaseTokenizerWrapper("t5-small")
model = Seq2SeqModelLoader("t5-small")

# Your experiments here...
```

## 📋 Checklist

- [ ] Initialize Git repository
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Create Colab notebook
- [ ] Clone repo in Colab
- [ ] Test import of your modules
- [ ] Set up sync workflow (git push/pull)

## 🎓 Example: Complete Workflow

### Local (VS Code):
```python
# llm_tokenizers/base_tokenizer.py
class BaseTokenizerWrapper:
    # ... your code ...
```

### Commit & Push:
```bash
git add llm_tokenizers/
git commit -m "Added new feature"
git push
```

### Colab:
```python
!git pull  # Get latest changes
from llm_tokenizers import BaseTokenizerWrapper
# Use your updated code!
```

This way, you get the best of both worlds:
- ✅ **Local**: Fast editing, good IDE support, version control
- ✅ **Colab**: Free GPU, cloud computing, easy sharing

