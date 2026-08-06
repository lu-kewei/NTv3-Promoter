
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from modelscope import AutoTokenizer

class iProMPDataset(Dataset):
    def __init__(self, data_path, dna_model_name, max_len=128):
        train_dataset = load_dataset("csv", 
                       data_files=data_path)["train"]
        self.sequences = train_dataset["sequence"]
        self.labels = train_dataset["label_num"]  
        self.tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
        self.max_length = max_len 

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        tokenized = self.tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(label, dtype=torch.long)
        return {
            "ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "targets": labels_tensor,
        }
        
    def __len__(self):
        return len(self.labels)
    

def load_iPro_dataset(data_path):
    train_dataset = load_dataset("csv", 
                       data_files=data_path)["train"]
    return train_dataset

def dna_collate_fn(
    batch,
    tokenizer,
    max_length: int = 2048,
):
    ref_sequences = [item["sequence"] for item in batch]
    labels = [item["label_num"] for item in batch]

    # Tokenize DNA sequences separately
    tokenized = tokenizer(
        ref_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    tokenized_batch = {
        "ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels_tensor,
    }

    return tokenized_batch
    
