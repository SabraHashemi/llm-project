"""
Setup script for Google Colab

Run this in a Colab cell to set up the project environment:
!wget https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/colab_setup.py
!python colab_setup.py

Or directly:
exec(open('colab_setup.py').read())
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "transformers>=4.30.0",
                          "torch>=2.0.0",
                          "matplotlib>=3.7.0",
                          "scikit-learn>=1.0.0",
                          "numpy>=1.24.0"])

def setup_project():
    """Set up project structure"""
    print("Setting up project structure...")
    
    # Create directories
    os.makedirs("llm_tokenizers", exist_ok=True)
    os.makedirs("llm_models", exist_ok=True)
    os.makedirs("examples", exist_ok=True)
    
    print("✅ Project structure created!")

if __name__ == "__main__":
    install_requirements()
    setup_project()
    print("\n✅ Setup complete!")
    print("Now you can use the modules in your Colab notebook!")

