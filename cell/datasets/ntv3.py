
import os 
from pathlib import Path
import torch 
from torch.utils.data import Dataset
import numpy as np 
import random 
from typing import Callable
import pandas as pd
from pyfaidx import Fasta
from transformers import  AutoTokenizer
import pyBigWig

def prepare_genomics_inputs(
    species: str,
    data_cache_dir: str | Path = "data",
    bigwig_file_ids: list[str] | None = None,
) -> tuple[str, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    """
    Downloads:
      1) FASTA from HF dataset under: <species>/genome.fasta
      2) BigWigs from HF dataset under: <species>/functional_tracks/**
         (filtered by bigwig_file_ids if provided)
      3) Splits from HF dataset under: <species>/splits.bed
      4) Metadata from HF dataset under: benchmark_metadata.tsv
    
    Args:
        species: Species name (e.g., "human", "arabidopsis")
        data_cache_dir: Directory where downloaded data files will be stored
        hf_repo_id: HuggingFace dataset repository ID
        bigwig_file_ids: Optional list of BigWig file IDs to download. If None,
            downloads all available BigWig files for the species.
    
    Returns:
      (fasta_path, bigwig_path_list, bigwig_file_ids)
    """
    cache = Path(data_cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    
    # --- Download metadata + <species> files (FASTA, BigWigs, Splits) ---
    metadata_file = "benchmark_metadata.tsv"
    download_patterns = [metadata_file, f"{species}/genome.fasta", f"{species}/splits.bed"]
    # Download all BigWig files
    download_patterns.append(f"{species}/functional_tracks/*.bigwig")
    # local_dir = Path(
    #     snapshot_download(
    #         repo_id=hf_repo_id,
    #         repo_type="dataset",
    #         allow_patterns=download_patterns,
    #         local_dir=str(cache),
    #     )
    # )
    local_dir = cache
    # --- Organize outputs ---
    # FASTA file
    fasta_path_repo = f"{species}/genome.fasta"
    fasta_path = str(local_dir / fasta_path_repo)
    
    # BigWig files - use downloaded files directly
    bigwig_dir = local_dir / species / "functional_tracks"
    
    if bigwig_file_ids is not None:
        bigwig_paths = [str(bigwig_dir / f"{file_id}.bigwig") for file_id in bigwig_file_ids]
        bigwig_ids = bigwig_file_ids
    else:
        # Find all downloaded BigWig files
        bigwig_paths = [str(bigwig_file) for bigwig_file in bigwig_dir.glob("*.bigwig")]
        bigwig_ids = [bigwig_file.stem for bigwig_file in bigwig_dir.glob("*.bigwig")]         
    
    # Splits file
    splits_path_repo = f"{species}/splits.bed"
    splits_path = local_dir / splits_path_repo

    splits_df = pd.read_csv(
        splits_path, 
        sep="\t", 
        header=None, 
        names=["chr_name", "start", "end", "split"],
        dtype={"chr_name": str, "start": int, "end": int, "split": str},
    )
    
    # Metadata file
    metadata_path = local_dir / metadata_file
    metadata_df = pd.read_csv(metadata_path, sep="\t")

    # Filter metadata according to species
    metadata_df = metadata_df[metadata_df["species_common_name"] == species].reset_index(drop=True)

    # Order metadata according to bigwig file ids
    metadata_df = (
      metadata_df.set_index("file_id")
        .loc[bigwig_ids]
        .reset_index()
    )

    return fasta_path, bigwig_paths, bigwig_ids, splits_df, metadata_df

def crop_center(x: np.ndarray, keep_target_center_fraction: float = 0.375) -> np.ndarray:
    """Crop the central sequence-length fraction for arrays of size (..., seq_len, num_tracks)"""
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]
_bigwig_cache = {}
def _get_bigwig_handle(bigwig_path: str) -> pyBigWig.pyBigWig:
    """Get or create a BigWig file handle for the current process."""
    
    process_id = os.getpid()
    abs_path = str(Path(bigwig_path).resolve())
    cache_key = (process_id, abs_path)
    
    if cache_key not in _bigwig_cache:
        # Check if file exists before trying to open
        if not Path(abs_path).exists():
            raise FileNotFoundError(
                f"BigWig file not found: {abs_path}\n"
                f"Original path: {bigwig_path}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        try:
            _bigwig_cache[cache_key] = pyBigWig.open(abs_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open BigWig file: {abs_path} with error: {str(e)}\n"
                f"File exists: {Path(abs_path).exists()}\n"
                f"File size: {Path(abs_path).stat().st_size if Path(abs_path).exists() else 'N/A'} bytes"
            ) from e
    
    return _bigwig_cache[cache_key]
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

class GenomeBigWigDataset(Dataset):
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
        bigwig_file_ids=None
    ):
        super().__init__()

        fasta_path, bigwig_paths, bigwig_ids, species_splits_df, metadata_df = prepare_genomics_inputs(
            species, 
            data_dir, 
            bigwig_file_ids
        )
        # Store paths instead of opening files immediately (for multi-worker compatibility)
        self.fasta_path = fasta_path
        self.bigwig_path_list = bigwig_paths
        self.sequence_length = sequence_length
        self.num_samples = num_samples 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.transform_fn =  create_targets_scaling_fn(metadata_df)
        self.keep_target_center_fraction = keep_target_center_fraction
        self.chrom_regions = species_splits_df
        # Filter regions by split
        split_regions = self.chrom_regions[self.chrom_regions["split"] == split].copy()

        # Filter valid regions (must be large enough for sequence_length)
        self.valid_regions = []
        for _, row in split_regions.iterrows():

            region_length = row.end - row.start
            if region_length < self.sequence_length:
                continue
            
            # Store valid region
            self.valid_regions.append((row.chr_name, row.start, row.end))

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
        tokenized = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"][0]  # Shape: (max_length,)

        # Signal from bigWig tracks (numpy array) -> torch tensor
        # Get BigWig handles lazily (cached per worker process)
        bigwig_targets = np.array([
            _get_bigwig_handle(bw_path).values(chrom, start, end, numpy=True)
            for bw_path in self.bigwig_path_list
        ])  # shape (num_tracks, seq_len)
        # Transpose to (seq_len, num_tracks)
        bigwig_targets = bigwig_targets.T
        # pyBigWig returns NaN where no data; turn NaN into 0
        bigwig_targets = torch.tensor(bigwig_targets, dtype=torch.float32)
        bigwig_targets = torch.nan_to_num(bigwig_targets, nan=0.0)
        
        # Crop targets to center fraction
        if self.keep_target_center_fraction < 1.0:
            bigwig_targets = crop_center(bigwig_targets, self.keep_target_center_fraction)

        # Apply scaling to targets
        bigwig_targets = self.transform_fn(bigwig_targets)

        sample = {
            "tokens": tokens,
            "targets": bigwig_targets,
            "chrom": chrom,
            "start": start,
            "end": end,
        }
        return sample