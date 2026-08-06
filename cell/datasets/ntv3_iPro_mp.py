import os
import os.path as osp 
import csv
from torch.utils.data import Dataset
import torch
from modelscope import AutoTokenizer

def _read_ipro_csv(data_path):
    sequences = []
    labels = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["sequence"])
            labels.append(int(row["label_num"]))
    return sequences, labels

class iProMPDataset(Dataset):
    def __init__(self, data_path, dna_model_name, max_len=128):
        if osp.isdir(data_path):
            sequences_list = []
            labels_list = []
            for data_name in os.listdir(data_path):
                if ".csv" not in data_name:
                    continue
                sequences, labels = _read_ipro_csv(osp.join(data_path, data_name))
                sequences_list.extend(sequences)
                labels_list.extend(labels)
            self.sequences = sequences_list
            self.labels = labels_list
        else:
            self.sequences, self.labels = _read_ipro_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
        self.max_length = max_len 

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        tokenized = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(label, dtype=torch.long)
        return {
            "tokens": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "targets": labels_tensor,
        }
        
    def __len__(self):
        return len(self.labels)
    

def load_iPro_dataset(data_path):
    sequences, labels = _read_ipro_csv(data_path)
    return {"sequence": sequences, "label_num": labels}

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
    
