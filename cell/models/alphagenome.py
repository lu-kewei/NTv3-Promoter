import torch 
from torch import nn 
import torch.nn.functional as F 
import os.path as osp 
import numpy as np 
from typing import Dict, List
from alphagenome_pytorch import AlphaGenome

def crop_center(x: np.ndarray, keep_target_center_fraction: float = 0.375) -> np.ndarray:
    """Crop the central sequence-length fraction for arrays of size (..., seq_len, num_tracks)"""
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]

class LinearHead(nn.Module):
    """A linear head that predicts one scalar value per track."""
    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.head(x)
        x = F.softplus(x)  # Ensure positive values
        return x
    
class ClassificationHead(nn.Module):
    """A linear head that predicts one scalar value per track."""
    def __init__(self, embed_dim: int, num_elements: int):
        super().__init__()
        self.num_elements = num_elements
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_elements*2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.head(x)
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(batch_size, sequence_length, self.num_elements, 2)
        return x

class HFModelWithHead(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head."""
    
    def __init__(
        self,
        model_name: str,
        head_names: List[str],
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()
        
        self.model = AlphaGenome()
        self.model.add_reference_heads('human')  # Add heads BEFORE loading weights
        # self.model.load_state_dict(torch.load(osp.join(model_name,"AlphaGenome_torch.pth")))
        self.model.load_from_official_jax_model(model_name)
        # torch.save(self.model.state_dict(),"AlphaGenome_torch.pth")
        self.keep_target_center_fraction = keep_target_center_fraction
        
        # self.head = LinearHead(embed_dim, len(class_names))
        self.bed_head = ClassificationHead(1536, len(head_names))
    
    def forward(self, tokens: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        tokens = tokens.squeeze(1)
        organism_torch = torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)
        outputs = self.model(tokens, organism_torch, return_embeds=True)
        embedding = outputs[0]  # Last hidden state
        
        # Crop to center fraction
        if self.keep_target_center_fraction < 1.0:
            embedding = crop_center(embedding, self.keep_target_center_fraction)
        
        logits = self.bed_head(embedding)
        
        return {"logits": logits}