from datasets import load_dataset


def load_math_dataset(dataset_name: str = "TIGER-Lab/VisualWebInstruct-Seed"):
    train_ds = load_dataset(dataset_name, "LongCoT", split="train[:500]")
    val_ds = load_dataset(dataset_name, "LongCoT", split="train[500:575]")
    test_ds = load_dataset(dataset_name, "LongCoT", split="train[575:650]")

    return train_ds, val_ds, test_ds
