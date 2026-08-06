from typing import Any, Dict, List, Tuple
import torch
def dna_collate_fn(
    batch: List[Dict[str, Any]],
    dna_tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Custom collate function for DNA models.
    在 DNA 序列分析（尤其是基因组学、变异检测）中，reference_sequence（参考序列）和 variant_sequence（变异序列）是两个核心概念，用于描述个体基因组与标准参考基因组之间的差异
    """
    ref_sequences = [item["reference_sequence"] for item in batch]
    alt_sequences = [item["variant_sequence"] for item in batch]

    # Tokenize DNA sequences separately
    tokenized_ref = dna_tokenizer(
        ref_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized_alt = dna_tokenizer(
        alt_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Get labels
    labels = []
    for item in batch:
        label = label2id[item["answer"]]
        labels.append(label)
    # Create labels tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    if "attention_mask" in tokenized_ref.keys():
        tokenized_batch = {
            "ref_ids": tokenized_ref.input_ids,
            "ref_attention_mask": tokenized_ref.attention_mask,
            "alt_ids": tokenized_alt.input_ids,
            "alt_attention_mask": tokenized_alt.attention_mask,
            "targets": labels_tensor,
        }
    else:
        tokenized_batch = {
            "ref_ids": tokenized_ref.input_ids,
            "alt_ids": tokenized_alt.input_ids,
            "targets": labels_tensor,
        }
    return tokenized_batch

def truncate_dna(
    example: Dict[str, Any], truncate_dna_per_side: int = 1024
) -> Dict[str, Any]:
    """
    Truncate DNA sequences by removing a specified number of base pairs from both ends.
    If the sequence is too short, it will return the middle portion.
    """
    for key in ["reference_sequence", "variant_sequence"]:
        sequence = example[key]
        seq_len = len(sequence)

        if seq_len > 2 * truncate_dna_per_side + 8:
            example[key] = sequence[truncate_dna_per_side:-truncate_dna_per_side]

    return example

def clean_variant_effect_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a variant effect example.
    """
    example['answer'] = example['answer'].split(";")[0].strip().lower()
    return example
