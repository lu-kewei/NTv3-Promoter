
import os 
import os.path as osp 
from pathlib import Path
import torch 
from torch.utils.data import Dataset
import numpy as np 
import random 
from typing import Callable
import pandas as pd
from pyfaidx import Fasta
# from transformers import  AutoTokenizer

import numpy as np
import torch

def dna_tokenizer(
    seq: str,
    padding="max_length",
    truncation=True,
    max_length=1024,
    return_tensors="pt",
):
    # A=0, C=1, G=2, T=3, N=0
    mapping = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0,
        'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 0
    }

    # encode
    ids = [mapping.get(c, 0) for c in seq]

    # truncation
    if truncation:
        ids = ids[:max_length]

    # padding
    if padding == "max_length":
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len

    input_ids = torch.tensor(ids, dtype=torch.long)

    if return_tensors == "pt":
        input_ids = input_ids.unsqueeze(0)  # (1, L)

    return {
        "input_ids": input_ids
    }

def prepare_genomics_inputs(species, data_cache_dir,class_names): 

    local_dir = data_cache_dir
   
    fasta_path_repo = osp.join(species,"genome.fasta")
    fasta_path = osp.join(local_dir, fasta_path_repo)

    bed_dir = osp.join(local_dir, species, "genome_annotation")

    bed_paths = [osp.join(bed_dir,class_name+".bed") for class_name in class_names]
    
    # Splits file
    splits_path_repo = osp.join(species,"splits.bed")
    splits_path = osp.join(local_dir, splits_path_repo)

    splits_df = pd.read_csv(
        splits_path, 
        sep="\t", 
        header=None, 
        names=["chr_name", "start", "end", "split"],
        dtype={"chr_name": str, "start": int, "end": int, "split": str},
    )
    
    return fasta_path, bed_paths, splits_df

def crop_center(x: np.ndarray, keep_target_center_fraction: float = 0.375) -> np.ndarray:
    """Crop the central sequence-length fraction for arrays of size (..., seq_len, num_tracks)"""
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]

_bed_cache = {}
def _get_bed_handle(bed_path: str) -> pd.DataFrame:
    """Get or create a Bed file handle for the current process."""
    process_id = os.getpid()
    abs_path = str(Path(bed_path).resolve())
    cache_key = (process_id, abs_path)
    if cache_key not in _bed_cache:
        # Check if file exists before trying to open
        if not Path(abs_path).exists():
            raise FileNotFoundError(f"Bed file not found: {abs_path}")
        
        try:
            _bed_cache[cache_key] = pd.read_csv(abs_path, sep="\t", header=None)
            _bed_cache[cache_key].columns = ["chr", "start", "end", "", "", "strand", "element"]
        except Exception as e:
            raise RuntimeError(f"Failed to open Bed file: {abs_path} with error: {str(e)}") from e
    
    return _bed_cache[cache_key]

_fasta_cache = {} 
def _get_fasta_handle(fasta_path: str) -> Fasta:
    """Get or create a FASTA file handle for the current process."""
   
    process_id = os.getpid()
    abs_path = str(Path(fasta_path).resolve())
    cache_key = (process_id, abs_path)
    
    if cache_key not in _fasta_cache:
        _fasta_cache[cache_key] = Fasta(abs_path, as_raw=True, sequence_always_upper=True)
    
    return _fasta_cache[cache_key]
def create_targets_scaling_fn(
    metadata_df: pd.DataFrame
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a scaling function that uses the track means to normalise and softclip the targets.
    """
    # Open bigwig files and compute track statistics
    track_means = metadata_df["mean"].to_numpy()
    print(f"Track means: {track_means}")
    print(f"Number of tracks: {track_means.shape}")

    # Create tensor from computed means
    track_means_tensor = torch.tensor(track_means, dtype=torch.float32)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        # Move constants to correct device then normalize
        means = track_means_tensor.to(x.device)
        scaled = x / means

        # Smooth clipping: if > 10, apply formula
        clipped = torch.where(
            scaled > 10.0,
            2.0 * torch.sqrt(scaled * 10.0) - 10.0,
            scaled,
        )
        return clipped

    return transform_fn

class GenomeBedDataset(Dataset):
    """
    A PyTorch dataset to access a reference genome and bigwig tracks. The dataset is 
    compatible with multi-worker DataLoaders (using process-local file handles and lazy 
    loading). For each sample, a random genomic region is picked from the specified split,
    and a random window of length `sequence_length` within that region is returned.
    """

    def __init__(
        self,
        data_dir,
        species,
        split: str,
        sequence_length: int,
        num_samples: int,
        model_name,
        keep_target_center_fraction: float = 1.0,
        class_names = None
    ):
        super().__init__()

        fasta_path, bed_paths, species_splits_df = prepare_genomics_inputs(
            species, 
            data_dir, 
            class_names
        )
        # Store paths instead of opening files immediately (for multi-worker compatibility)
        self.fasta_path = fasta_path
        self.bed_path_list = bed_paths
        self.sequence_length = sequence_length
        self.num_samples = num_samples 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # self.transform_fn =  create_targets_scaling_fn(metadata_df)
        self.keep_target_center_fraction = keep_target_center_fraction
        self.chrom_regions = species_splits_df
        # Filter regions by split
        split_regions = self.chrom_regions[self.chrom_regions["split"] == split].copy()

        # Filter valid regions (must be large enough for sequence_length)
        self.valid_regions = [
            (r.chr_name, r.start, r.end) 
            for r in split_regions.itertuples() 
            if r.end - r.start >= self.sequence_length
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample a random region from the valid regions
        chrom, region_start, region_end = random.choice(self.valid_regions)
        
        # Sample a random window within this region
        max_start = region_end - self.sequence_length
        start = random.randint(region_start, max_start)
        end = start + self.sequence_length

        # Sequence - get FASTA handle lazily (cached per worker process)
        fasta = _get_fasta_handle(self.fasta_path)
        seq = fasta[chrom][start:end]  # string slice
        # Tokenize with padding and truncation to ensure consistent lengths for batching
        # tokenized = self.tokenizer(
        #     seq,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.sequence_length,
        #     return_tensors="pt",
        # )
        # tokens = tokenized["input_ids"][0]  # Shape: (max_length,)
        tokenized = dna_tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"]  # shape: (1, L)

        # Get bed targets
        bed_sequence_length = self.sequence_length * self.keep_target_center_fraction
        bed_start = int(start + (self.sequence_length - bed_sequence_length) // 2)
        bed_end = int(bed_start + bed_sequence_length)
        bed_targets = np.zeros((int(bed_sequence_length), len(self.bed_path_list)), dtype=np.int32)
        for bed_idx, bed_path in enumerate(self.bed_path_list):
            bed_df = _get_bed_handle(bed_path)
            regions = bed_df[(bed_df["chr"] == chrom) & (bed_df["start"] >= bed_start) & (bed_df["end"] <= bed_end)]
            for _, row in regions.iterrows():
                bed_targets[row["start"] - bed_start:row["end"] - bed_start, bed_idx] = 1

        # pyBigWig returns NaN where no data; turn NaN into 0
        bed_targets = torch.tensor(bed_targets, dtype=torch.int64)

        sample = {
            "tokens": tokens,
            "targets": bed_targets,
            "chrom": chrom,
            "start": start,
            "end": end,
        }
        return sample