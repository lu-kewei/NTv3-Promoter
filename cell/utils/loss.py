
import torch
from torch import nn 
import torch.nn.functional as F

def poisson_loss(ytrue: torch.Tensor, ypred: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Poisson loss per element: ypred - ytrue * log(ypred)."""
    return ypred - ytrue * torch.log(ypred + epsilon)

def safe_for_grad_log_torch(x: torch.Tensor) -> torch.Tensor:
    """Guarantees that the log is defined for all x > 0 in a differentiable way."""
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Computes focal loss for nucleotide-level classification tasks from logits.
    It handles masking of invalid positions. Includes optional class weights.

    """
    # Compute probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probs)

    # Reshape for loss computation
    # num_classes: scalar
    num_classes = probabilities.shape[-1]
    probabilities = torch.reshape(probabilities, (-1, num_classes))
    log_probs = torch.reshape(log_probs, (-1, num_classes))
    targets = torch.reshape(targets, (-1,))


    # Compute focal loss per position
    loss = -torch.sum(
        torch.gather(
            (1 - probabilities) ** gamma * log_probs,
            dim=-1,
            index=targets[..., None],
        ),
        dim=-1,
    )  # shape: (total_positions,)

    # Average loss over valid positions only
    loss = loss.sum() / (loss.numel() + epsilon)  # type: ignore

    return loss

def poisson_multinomial_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    shape_loss_coefficient: float = 5.0,
    epsilon: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Regression loss for bigwig tracks (Poisson-Multinomial). The logits and targets are
    expected to be of shape (batch, seq_length, num_head).
    """
    batch_size, seq_length, num_head = logits.shape
    
    # Scale loss: Poisson loss on total counts per sequence per track
    # Sum over sequence dimension (axis=1)
    sum_pred = logits.sum(dim=1)  # (batch, num_head)
    sum_true = targets.sum(dim=1)  # (batch, num_head)
    
    # Compute poisson loss per (batch, track)
    scale_loss = poisson_loss(sum_true, sum_pred, epsilon=epsilon)  # (batch, num_head)
    
    # Normalize by sequence length
    scale_loss = scale_loss / (seq_length + epsilon)
    
    # Average over batch and tracks
    scale_loss = scale_loss.mean()
    
    # Shape loss: Multinomial loss
    # Add epsilon to all positions
    predicted_counts = logits + epsilon
    targets_with_epsilon = targets + epsilon
    
    # Normalize predictions to get probabilities
    denom = predicted_counts.sum(dim=1, keepdim=True) + epsilon  # (batch, 1, num_head)
    p_pred = predicted_counts / denom
    
    # Compute shape loss: -sum(targets * log(p_pred))
    pl_pred = safe_for_grad_log_torch(p_pred)
    shape_loss = -(targets_with_epsilon * pl_pred)
    
    # Sum over all dimensions and normalize by total number of positions
    shape_denom = batch_size * seq_length * num_head + epsilon
    shape_loss = shape_loss.sum() / shape_denom
    
    # Combine losses
    loss = shape_loss + scale_loss / shape_loss_coefficient

    return loss

class FocalLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self,logits, targets):
        return focal_loss(logits,targets)
class PoissonMultinomialLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self,logits, targets):
        return poisson_multinomial_loss(logits,targets)
    
def Criterion(kwargs):
    if kwargs["type"] == "entropy_loss":
        return  torch.nn.CrossEntropyLoss()
    elif kwargs["type"] == "focal_loss":
        return FocalLoss()
    elif kwargs["type"] == "poisson_loss":
        return PoissonMultinomialLoss()
