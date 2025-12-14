from datasets import load_dataset

ds = load_dataset("readerbench/ro-offense-sequences")

print("here")

print(ds.shape)

print(ds["train"][0])