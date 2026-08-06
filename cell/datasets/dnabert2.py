import csv 
import torch
from torch.utils.data import Dataset
from modelscope import AutoTokenizer

class DNADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 model_name,
                 data_path,
                 max_len=128
                ):

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            print("Perform single sequence classification...")
            self.texts = [d[0] for d in data]
            self.labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            print("Perform sequence-pair classification...")
            self.texts = [[d[0], d[1]] for d in data]
            self.labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_len 

    def __getitem__(self, idx):
        sequence = self.texts[idx]
        label = self.labels[idx]

        tokenized = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(label, dtype=torch.long)
        return {
            "tokens": tokenized["input_ids"][0],
            "targets": labels_tensor,
        }
        
    def __len__(self):
        return len(self.labels)