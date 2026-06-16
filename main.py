from datasets import load_dataset

ds = load_dataset("webis/tldr-17", revision="refs/convert/parquet")

print(ds['train'][0])
