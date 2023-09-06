from sys import argv,exit
import os
from huggingface_hub import snapshot_download

if len(argv) < 3:
   print(f'Usage: python {argv[0]} <repo_name> <dir_name_in_models>')
   exit(1)
repo_name = argv[1]
dir_name = './models/'+argv[2]
HF_API_KEY=os.environ["HF_API_KEY"]
dld_dir = snapshot_download(repo_id=repo_name,local_dir=dir_name,local_dir_use_symlinks=False,
    token=HF_API_KEY, max_workers=1, ignore_patterns="pytorch*.bin")
print(f'Downloaded model to {dld_dir}!')

