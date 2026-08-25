from huggingface_hub import list_repo_files
for f in list_repo_files("Luckyjhg/Geo170K", repo_type="dataset"):
    print(f)