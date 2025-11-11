"""
Setup helpers for Google Colab (optional).
"""

import subprocess
import sys
import os


def install_requirements() -> None:
\tprint("Installing requirements...")
\tsubprocess.check_call([
\t\tsys.executable, "-m", "pip", "install", "-q",
\t\t"transformers>=4.30.0",
\t\t"torch>=2.0.0",
\t\t"matplotlib>=3.7.0",
\t\t"scikit-learn>=1.0.0",
\t\t"numpy>=1.24.0",
\t\t"python-dateutil>=2.8.0",
\t])


def setup_project() -> None:
\tprint("Setting up project structure...")
\tos.makedirs("examples", exist_ok=True)
\tprint("✅ Project structure checked.")


if __name__ == "__main__":
\tinstall_requirements()
\tsetup_project()
\tprint("\\n✅ Setup complete! You can now run the examples and tests.")


