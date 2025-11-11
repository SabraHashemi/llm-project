# Fix: Colab Git Clone Authentication Issue

If cloning fails for a private repo, either make it public or use a token:

```python
from getpass import getpass
token = getpass("GitHub Token (repo:read): ")
!git clone https://{token}@github.com/SabraHashemi/llm-project.git
%cd llm-project
```


