from datasets import load_dataset

# Download dataset
train_ds = load_dataset("TIGER-Lab/VisualWebInstruct-Seed", "LongCoT", split="train[:500]")
val_ds = load_dataset("TIGER-Lab/VisualWebInstruct-Seed", "LongCoT", split="train[500:575]")
test_ds = load_dataset("TIGER-Lab/VisualWebInstruct-Seed", "LongCoT", split="train[575:650]")