from torch.utils.data import Dataset
from PIL import Image

from data.preprocess_data import prepare_cot_dataset


class MathReasoningDataset(Dataset):
    def __init__(self, data_list, processor, max_length=1024):
        self.data_list = data_list
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        messages = prepare_cot_dataset(item)
        text = self.processor.apply_chat_template(messages, tokenize=False)

        # The processor handles tokenizing text and extracting image patches
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # For causal LM, labels are usually the input_ids, padded tokens set to -100
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return {k: v.squeeze(0) for k, v in inputs.items()}
