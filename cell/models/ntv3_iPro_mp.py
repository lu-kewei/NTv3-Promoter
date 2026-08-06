import torch 
from torch import nn 
import torch.nn.functional as F 
from transformers import AutoConfig, AutoModelForMaskedLM
import numpy as np 
from typing import Dict, List

POOLING_TYPES = (
    "masked_mean",
    "masked_max",
    "query_attention_1head",
    "query_attention_8head",
)

class PoolingModule(nn.Module):
    """PAD-aware pooling with a common ``[B, D]`` output contract."""

    def __init__(self, hidden_size: int, pooling_type: str = "query_attention_8head"):
        super().__init__()
        if pooling_type not in POOLING_TYPES:
            raise ValueError(
                f"Unknown pooling_type={pooling_type!r}; expected one of {POOLING_TYPES}"
            )
        self.hidden_size = hidden_size
        self.pooling_type = pooling_type
        if pooling_type.startswith("query_attention_"):
            num_heads = 1 if pooling_type.endswith("1head") else 8
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, embeddings, attention_mask=None, return_attention=False):
        if embeddings.ndim != 3:
            raise ValueError(f"hidden states must have shape [B, L, D], got {embeddings.shape}")
        if attention_mask is None:
            valid = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        else:
            if attention_mask.shape != embeddings.shape[:2]:
                raise ValueError(
                    f"attention_mask shape {attention_mask.shape} does not match "
                    f"hidden states {embeddings.shape[:2]}"
                )
            valid = attention_mask.to(device=embeddings.device, dtype=torch.bool)
        if (~valid).all(dim=1).any():
            raise ValueError("pooling received an all-PAD sequence")

        if self.pooling_type == "masked_mean":
            mask = valid.unsqueeze(-1).to(dtype=embeddings.dtype)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return (pooled, None) if return_attention else pooled
        if self.pooling_type == "masked_max":
            masked = embeddings.masked_fill(~valid.unsqueeze(-1), torch.finfo(embeddings.dtype).min)
            pooled = masked.max(dim=1).values
            return (pooled, None) if return_attention else pooled

        batch_size = embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        context, attention_weights = self.attention(
            query=query,                  # [batch_size, 1, hidden_size]
            key=embeddings,               # [batch_size, seq_len, hidden_size]
            value=embeddings,             # [batch_size, seq_len, hidden_size]
            key_padding_mask=~valid,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        
        # Squeeze out the singleton dimension
        pooled = context.squeeze(1)       # [batch_size, hidden_size]
        if not return_attention:
            return pooled
        # Preserve the native per-head/query dimensions for auditability:
        # [batch, heads, query_length=1, sequence_length].
        return pooled, attention_weights


class SelfAttentionPooling(PoolingModule):
    """Backward-compatible name for the original eight-head implementation."""

    def __init__(self, hidden_size, num_heads=8):
        if num_heads not in (1, 8):
            raise ValueError("SelfAttentionPooling supports only 1 or 8 heads")
        super().__init__(hidden_size, f"query_attention_{num_heads}head")
    
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
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.head(x)
        return x
 
class HFModelWithHead(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head."""
    
    def __init__(
        self,
        model_name: str,
        head_names: List[str],
        pooling_type: str = "query_attention_8head",
    ):
        super().__init__()
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.config,
        )
        # self.backbone = torch.compile(backbone)
        self.backbone = backbone
        
        # self.head = LinearHead(embed_dim, len(class_names))
        self.bed_head = ClassificationHead(self.config.embed_dim, 2)
        self.model_name = model_name
        self.pool = PoolingModule(self.config.embed_dim, pooling_type=pooling_type)
        self.pooling_type = pooling_type
    
    def forward(self, batch, return_features: bool = False) -> Dict[str, torch.Tensor]:
        # Forward through backbone
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = batch["tokens"].ne(self.config.pad_token_id)
        outputs = self.backbone(
            input_ids=batch["tokens"],
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        token_embeddings = outputs.hidden_states[-1]
        if not return_features:
            pooled_embedding = self.pool(
                token_embeddings,
                attention_mask=attention_mask,
            )
            return self.bed_head(pooled_embedding)

        pooled_embedding, attention_per_head = self.pool(
            token_embeddings,
            attention_mask=attention_mask,
            return_attention=True,
        )
        sequence_embedding = self.bed_head.layer_norm(pooled_embedding)
        logits = self.bed_head.head(sequence_embedding)
        if attention_per_head is None:
            attention_mean = None
        else:
            attention_mean = attention_per_head.mean(dim=1).squeeze(1)
        return {
            "logits": logits,
            "sequence_embedding": sequence_embedding,
            # Average the eight heads first, then remove only the singleton
            # query dimension. No second softmax or renormalization is applied.
            "attention_mean": attention_mean,
            "attention_per_head": attention_per_head,
            "attention_mask": attention_mask,
        }
