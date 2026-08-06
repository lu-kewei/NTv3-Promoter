from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.pipelines import Pipeline

try:
    from pyfaidx import Fasta
except Exception:
    Fasta = None

try:
    import requests
except Exception:
    requests = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


# ---------------------------------------------------------------------
# Assembly <-> species mapping
# ---------------------------------------------------------------------
ASSEMBLY_TO_SPECIES = {
    "hg38": "human",
    "mm10": "mouse",
    "dm6": "drosophila_melanogaster",
    "TAIR10": "arabidopsis_thaliana",
    "Zm-B73-REFERENCE-NAM-5.0": "zea_mays",
    "IRGSP-1.0": "oryza_sativa",
    "Glycine_max_v2.1": "glycine_max",
    "IWGSC": "triticum_aestivum",
    "Gossypium_hirsutum_v2.1": "gossypium_hirsutum",
    "ASM228892v3": "delphinapterus_leucas",
    "ASM334442v1": "ursus_americanus",
    "AmpOce1": "amphiprion_ocellaris",
    "Bison_UMD1": "bison_bison_bison",
    "ChiLan1": "chinchilla_lanigera",
    "Felis_catus_9": "felis_catus",
    "GRCz11": "danio_rerio",
    "KH": "ciona_intestinalis",
    "Mnem_1": "macaca_nemestrina",
    "R64": "saccharomyces_cerevisiae",
    "ROS_Cfam_1": "canis_lupus_familiaris",
    "SCA1": "serinus_canaria",
    "TETRAODON8": "tetraodon_nigroviridis",
    "WBcel235": "caenorhabditis_elegans",
    "bGalGal1": "gallus_gallus",
    "fSalTru1": "salmo_trutta",
    "gorGor4": "gorilla_gorilla",
    "mRatBN7": "rattus_norvegicus",
    "SL3": "solanum_lycopersicum",
    "ARS-UCD2.0": "bos_taurus",
}
SPECIES_TO_ASSEMBLY = {v: k for k, v in ASSEMBLY_TO_SPECIES.items()}

# Minimal UCSC FASTA sources (extend as needed)
ASSEMBLY_TO_UCSC_FA_GZ = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    "dm6":  "https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz",
}


def _sanitize_dna(seq: str) -> str:
    seq = seq.upper()
    return "".join(ch if ch in ("A", "C", "G", "T", "N") else "N" for ch in seq)


def _download_file(url: str, dst: Path) -> None:
    if requests is None:
        raise ImportError("requests is required for genome download. Install with: pip install requests")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _fetch_from_api(assembly: str, chrom: str, start: int, end: int) -> str:
    """
    Fetch DNA sequence from UCSC API.
    
    Parameters
    ----------
    assembly : str
        Genome assembly name (e.g., "hg38", "mm10")
    chrom : str
        Chromosome name
    start : int
        Start position (0-based)
    end : int
        End position (0-based, exclusive)
    
    Returns
    -------
    str
        DNA sequence (uppercase, sanitized)
    """
    # if requests is None:
    #     raise ImportError("requests is required for API fetching. Install with: pip install requests")
    
    # url = f"https://api.genome.ucsc.edu/getData/sequence?genome={assembly};chrom={chrom};start={start};end={end}"
    # response = requests.get(url)
    # response.raise_for_status()
    # seq = response.json()["dna"].upper()

    import pysam

    fasta_path = "/home/lukewei/datasets/cell/NTv3/benchmark/human/genome.fasta"

    def get_seq_local(assembly, chrom, start, end):
        # assembly 参数可保留用于兼容，但本地不一定需要
        with pysam.FastaFile(fasta_path) as fa:
            seq = fa.fetch(chrom, start, end).upper()
        return seq
    seq = get_seq_local(assembly, chrom, start, end)
    
    return _sanitize_dna(seq)


def _pick_device(device: Union[str, int, torch.device]) -> torch.device:
    # Handle torch.device objects
    if isinstance(device, torch.device):
        return device
    
    # Handle integer device IDs (transformers pipeline convention)
    if isinstance(device, int):
        if device == -1:
            return torch.device("cpu")
        elif device >= 0:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            else:
                return torch.device("cpu")
        else:
            raise ValueError(f"Invalid device integer: {device}")
    
    # Handle string device names
    if isinstance(device, str):
        d = device.lower()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if d in ("cuda", "cpu", "mps"):
            return torch.device(d)
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda', 'mps', or an integer")
    
    raise ValueError(f"device must be a string, integer, or torch.device, got {type(device)}")


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def _plot_tracks_fillbetween(
    tracks: Dict[str, np.ndarray],
    chrom: Optional[str],
    start: int,
    end: int,
    assembly: Optional[str],
    height: float = 1.0,
    figsize_x: float = 20.0,
):
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    if sns is None:
        raise ImportError("seaborn is required for notebook-style plots. Install with: pip install seaborn")

    n = len(tracks)
    if n == 0:
        raise ValueError("No tracks to plot.")

    fig, axes = plt.subplots(n, 1, figsize=(figsize_x, height * n), sharex=True)
    if n == 1:
        axes = [axes]

    any_track = next(iter(tracks.values()))
    x = np.linspace(start, end, num=len(any_track), endpoint=False)

    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(x, y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)

    label = f"{chrom}:{start}-{end}" if chrom is not None else f"{start}-{end}"
    if assembly is not None:
        label += f" ({assembly})"
    axes[-1].set_xlabel(label)

    plt.tight_layout()
    return fig, axes


@dataclass
class NTv3TracksOutput:
    bigwig_tracks_logits: np.ndarray  # (L_pred, T)
    bed_tracks_logits: np.ndarray     # (L_pred, E, C)
    mlm_logits: np.ndarray
    chrom: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    species: Optional[str] = None
    assembly: Optional[str] = None
    bigwig_track_names: Optional[List[str]] = None  # from cfg.bigwigs_per_species[species]
    bed_element_names: Optional[List[str]] = None
    window_len: Optional[int] = None
    pred_start: Optional[int] = None
    pred_end: Optional[int] = None


class NTv3TracksPipeline(Pipeline):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        tokenizer: Optional[Union[str, Any]] = None,
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        default_species: str = "human",
        genome_cache_dir: Union[str, Path] = "~/.cache/ntv3/genomes",
        device: str = "auto",
        mps_force_cpu: bool = True,
        mps_force_cpu_length: int = 16384,
        verbose: bool = True,
        # Your notebook uses these constants for "middle 37.5%" prediction span
        pred_center_fraction: float = 0.375,
        pred_center_offset_fraction: float = 0.3125,
        **kwargs: Any,
    ):
        self.model_id = model if isinstance(model, str) else None
        self.default_species = default_species
        self.genome_cache_dir = Path(genome_cache_dir)
        self.mps_force_cpu = bool(mps_force_cpu)
        self.mps_force_cpu_length = int(mps_force_cpu_length)
        self.verbose = bool(verbose)
        self.pred_center_fraction = float(pred_center_fraction)
        self.pred_center_offset_fraction = float(pred_center_offset_fraction)

        if self.default_species not in SPECIES_TO_ASSEMBLY:
            raise ValueError(
                f"default_species='{self.default_species}' is not supported. "
                f"Supported species: {sorted(SPECIES_TO_ASSEMBLY.keys())}"
            )

        if isinstance(model, str):
            self.config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, token=token)
            self.model = AutoModel.from_pretrained(model, trust_remote_code=trust_remote_code, token=token)
        else:
            self.model = model
            self.config = getattr(model, "config", None)

        if tokenizer is None:
            if not self.model_id:
                raise ValueError("If passing a model module, pass tokenizer explicitly.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust_remote_code, token=token)
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code, token=token)
        else:
            self.tokenizer = tokenizer

        # Extract model_id from config if not already set (following ntv3_gff_pipeline.py pattern)
        if self.model_id is None and self.config is not None:
            self.model_id = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)

        # bed names (your notebooks refer to bed_element_names)
        self.bed_element_names = (
            getattr(self.config, "bed_elements_names", None)
            or getattr(self.config, "bed_element_names", None)
        )

        self._target_device = _pick_device(device)
        self.model.to(self._target_device)
        self.model.eval()

        super().__init__(model=self.model, tokenizer=self.tokenizer, device=-1, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def _get_model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _resolve_species_and_assembly(self, inputs: Dict[str, Any]) -> tuple[str, str]:
        species = inputs.get("species", self.default_species)
        if species not in SPECIES_TO_ASSEMBLY:
            raise ValueError(f"Unsupported species='{species}'. Supported species: {sorted(SPECIES_TO_ASSEMBLY.keys())}")
        assembly = SPECIES_TO_ASSEMBLY[species]
        return species, assembly


    def _maybe_force_cpu_for_mps_long(self, input_ids_cpu: torch.Tensor) -> torch.device:
        dev = self._get_model_device()
        if self.mps_force_cpu and dev.type == "mps":
            seq_len = int(input_ids_cpu.shape[-1])
            if seq_len >= self.mps_force_cpu_length:
                if self.verbose:
                    print(
                        f"[NTv3TracksPipeline] MPS detected and input is long (tokens={seq_len}). "
                        "Switching model + inputs to CPU for this run."
                    )
                self.model.to("cpu")
                self.model.eval()
                return torch.device("cpu")
        return dev

    def preprocess(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        species, assembly = self._resolve_species_and_assembly(inputs)

        # Resolve sequence
        if "seq" in inputs and inputs["seq"] is not None:
            seq = _sanitize_dna(inputs["seq"])
            chrom = None
            start = 0
            end = len(seq)
            window_len = len(seq)
        else:
            chrom = inputs["chrom"]
            start = int(inputs["start"])
            end = int(inputs["end"])
            window_len = end - start
            # Fetch sequence from UCSC API instead of downloading genome
            seq = _fetch_from_api(assembly, chrom, start, end)

        # Tokenize with padding
        batch = self.tokenizer([seq], add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")
        input_ids_cpu = batch["input_ids"]

        # MPS-long fallback decision
        device = self._maybe_force_cpu_for_mps_long(input_ids_cpu)

        # Move inputs
        input_ids = input_ids_cpu.to(device)
        # Species tokenization - match batch size
        batch_size = input_ids.shape[0]
        species_ids = self.model.encode_species([species] * batch_size)
        species_ids_tensor = species_ids.to(device)

        # Prediction interval (not used for slicing logits, just x-axis)
        pred_start = start + int(window_len * self.pred_center_offset_fraction)
        pred_end = pred_start + int(window_len * self.pred_center_fraction)

        # ✅ The source of truth for track IDs/names (your note)
        bigwig_track_names = list(self.config.bigwigs_per_species[species])

        return {
            "input_ids": input_ids,
            "species_ids": species_ids_tensor,
            "meta": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "species": species,
                "assembly": assembly,
                "window_len": window_len,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "bigwig_track_names": bigwig_track_names,
            },
        }

    # prevent Pipeline from moving tensors to its own device
    def forward(self, model_inputs, **forward_params):
        return self._forward(model_inputs, **forward_params)

    def _forward(self, model_inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        meta = model_inputs.pop("meta")
        if self.verbose:
            print(f"Running on device: {self._get_model_device()}")
        with torch.no_grad():
            out = self.model(
                input_ids=model_inputs["input_ids"],
                species_ids=model_inputs["species_ids"],
            )
        # Return a dict containing the model output and meta separately
        # since ModelOutput objects are immutable
        return {"model_output": out, "meta": meta}

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs: Any) -> NTv3TracksOutput:
        # Extract model_output and meta from the dict returned by _forward
        if isinstance(model_outputs, dict) and "model_output" in model_outputs:
            model_out = model_outputs["model_output"]
            meta = model_outputs.get("meta", {})
        else:
            # Fallback for direct ModelOutput (shouldn't happen with current code)
            model_out = model_outputs
            meta = {}

        def to_np(x):
            return x.detach().float().cpu().numpy()

        bigwig_np = to_np(model_out["bigwig_tracks_logits"])
        bed_np = to_np(model_out["bed_tracks_logits"])
        mlm_np = to_np(model_out["logits"])

        # Normalize shapes to remove batch/(optional assembly) dims
        if bigwig_np.ndim == 3:
            bigwig_np = bigwig_np[0]          # (L, T)
        elif bigwig_np.ndim == 4:
            bigwig_np = bigwig_np[0, 0]       # (L, T) if (B, A, L, T)
        else:
            raise ValueError(f"Unexpected bigwig_tracks_logits ndim: {bigwig_np.ndim}")

        if bed_np.ndim == 4:
            bed_np = bed_np[0]                # (L, E, C)
        elif bed_np.ndim == 5:
            bed_np = bed_np[0, 0]             # (L, E, C) if (B, A, L, E, C)
        else:
            raise ValueError(f"Unexpected bed_tracks_logits ndim: {bed_np.ndim}")

        if mlm_np.ndim == 3:
            mlm_np = mlm_np[0]

        return NTv3TracksOutput(
            bigwig_tracks_logits=bigwig_np,
            bed_tracks_logits=bed_np,
            mlm_logits=mlm_np,
            chrom=meta.get("chrom"),
            start=meta.get("start"),
            end=meta.get("end"),
            species=meta.get("species"),
            assembly=meta.get("assembly"),
            bigwig_track_names=meta.get("bigwig_track_names"),
            bed_element_names=self.bed_element_names,
            window_len=meta.get("window_len"),
            pred_start=meta.get("pred_start"),
            pred_end=meta.get("pred_end"),
        )

    def __call__(
        self,
        inputs,
        *args,
        plot: bool = False,
        tracks_to_plot: Optional[Dict[str, str]] = None,   # title -> track_id (ENCSR...)
        elements_to_plot: Optional[List[str]] = None,       # element names
        plot_height: float = 1.0,
        plot_figsize_x: float = 20.0,
        **kwargs,
    ):
        """
        One-step call that can optionally plot and always returns NTv3TracksOutput.
        """
        out: NTv3TracksOutput = super().__call__(inputs, *args, **kwargs)

        if plot:
            if out.bigwig_track_names is None:
                raise ValueError("bigwig_track_names missing; expected cfg.bigwigs_per_species[species].")
            if out.bed_element_names is None:
                raise ValueError("bed element names missing from config.")
            tracks_to_plot = tracks_to_plot or {}
            elements_to_plot = elements_to_plot or []

            bigwig_names = out.bigwig_track_names
            bed_element_names = out.bed_element_names

            # Validate
            missing_tracks = [tid for tid in tracks_to_plot.values() if tid not in bigwig_names]
            if missing_tracks:
                raise ValueError(
                    f"The following tracks are not available in bigwig_names: {missing_tracks}\n"
                    f"First 50 available: {bigwig_names[:50]}{'...' if len(bigwig_names) > 50 else ''}"
                )

            missing_elements = [e for e in elements_to_plot if e not in bed_element_names]
            if missing_elements:
                raise ValueError(
                    f"The following elements are not available in bed_element_names: {missing_elements}\n"
                    f"First 50 available: {bed_element_names[:50]}{'...' if len(bed_element_names) > 50 else ''}"
                )

            # Build bigwig tracks dict (title -> y)
            bigwig_tracks: Dict[str, np.ndarray] = {}
            bigwig = out.bigwig_tracks_logits  # (L_pred, T)
            for title, track_id in tracks_to_plot.items():
                track_idx = bigwig_names.index(track_id)
                bigwig_tracks[title] = bigwig[:, track_idx]

            # Bed positive class probabilities (title -> y)
            bed_probs: Dict[str, np.ndarray] = {}
            probs = _softmax_last(out.bed_tracks_logits)  # (L_pred, E, C)
            for element_name in elements_to_plot:
                element_idx = bed_element_names.index(element_name)
                bed_probs[element_name] = probs[:, element_idx, 1]

            all_tracks = {**bigwig_tracks, **bed_probs}

            plot_start = int(out.pred_start or 0)
            plot_end = int(out.pred_end or (plot_start + len(next(iter(all_tracks.values())))))
            
            _plot_tracks_fillbetween(
                all_tracks,
                chrom=out.chrom,
                start=plot_start,
                end=plot_end,
                assembly=out.assembly,
                height=plot_height,
                figsize_x=plot_figsize_x,
            )
        return out

def load_ntv3_tracks_pipeline(
    model: str,
    device: str = "auto",
    **pipeline_kwargs: Any,
):
    """
    Convenience helper to build an NTv3TracksPipeline for any NTv3 checkpoint.
    Parameters
    ----------
    model:
        Checkpoint id, e.g. "InstaDeepAI/NTv3_100M", "InstaDeepAI/NTv3_650M", ...
    device:
        "auto", "cpu", "cuda", "mps"
    pipeline_kwargs:
        Extra kwargs passed to NTv3TracksPipeline (default_species, genome_cache_dir, etc.).
    """
    pipe = NTv3TracksPipeline(
        model=model,
        trust_remote_code=True,
        device=device,
        **pipeline_kwargs,
    )
    return pipe