"""
Quick Start Code for Colab - Copy and Paste This!
"""

# Option 1: Public repo
# !git clone https://github.com/SabraHashemi/llm-project.git
# %cd llm-project

# Option 2: Private repo (token)
# from getpass import getpass
# token = getpass("GitHub Token (repo:read): ")
# !git clone https://{token}@github.com/SabraHashemi/llm-project.git
# %cd llm-project

# Install deps
# !pip install -q transformers torch matplotlib scikit-learn numpy python-dateutil

import sys
sys.path.insert(0, '.')  # or '/content' if uploaded

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader

print("✅ Setup complete!")


