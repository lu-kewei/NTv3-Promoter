from cell.utils.data import clean_variant_effect_example,truncate_dna
from datasets import load_dataset

def load_variant_effect_dataset(data_dir,truncate_dna_per_side):
    # elif args.dataset_type == "variant_effect_coding":
    dataset = load_dataset(data_dir)
    dataset = dataset.map(clean_variant_effect_example)

    if truncate_dna_per_side:
        dataset = dataset.map(
            truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side}
        )

    labels = []
    for split, data in dataset.items():
        labels.extend(data["answer"])
    labels = sorted(list(set(labels)))
    
    print(f"Dataset:\n{dataset}\nLabels:\n{labels}\nNumber of labels:{len(labels)}")
    return dataset, labels

def load_variant_effect_non_snv_dataset(data_dir,truncate_dna_per_side):
    # elif args.dataset_type == "variant_effect_coding":
    dataset = load_dataset(data_dir)
    dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
    dataset = dataset.map(clean_variant_effect_example)

    if truncate_dna_per_side:
        dataset = dataset.map(
            truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side}
        )

    labels = []
    for split, data in dataset.items():
        labels.extend(data["answer"])
    labels = sorted(list(set(labels)))
    
    print(f"Dataset:\n{dataset}\nLabels:\n{labels}\nNumber of labels:{len(labels)}")
    return dataset, labels