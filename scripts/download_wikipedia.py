from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(dir(ds))
ds.save_to_disk("wikipedia-dataset")
