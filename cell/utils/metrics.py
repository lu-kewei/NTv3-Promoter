import torch
from torch import nn 
from torchmetrics import PearsonCorrCoef
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    MulticlassMatthewsCorrCoef,
)

class GlobalPearsonCorrCoef(Metric):
    """
    preds:   (B, L, C)
    target:  (B, L, C)

    Compute global Pearson correlation over (B * L, C).
    """

    def __init__(self,num_tracks):
        super().__init__()
        self.pcc = PearsonCorrCoef(num_outputs = num_tracks)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        if preds.ndim != 3:
            raise ValueError(f"Expected (B, L, C), got {preds.shape}")

        B, L, C = preds.shape

        preds_flat = preds.reshape(B * L, C).to(torch.float64)
        target_flat = target.reshape(B * L, C).to(torch.float64)
        # pred_flat = predictions.detach().reshape(-1, self.num_tracks)  # (N, num_tracks)
        # target_flat = targets.detach().reshape(-1, self.num_tracks).to(torch.float64)

        self.pcc.update(preds_flat, target_flat)

    def compute(self):
        return self.pcc.compute()

    def reset(self):
        self.pcc.reset()
        
class MultiHeadMulticlassMCC(Metric):
    """
    Compute MCC for preds of shape (N, C, K),
    target of shape (N, C),
    then average MCC over C dimension.
    """

    def __init__(self, num_classes: int, num_heads: int):
        super().__init__()

        self.num_classes = num_classes
        self.num_heads = num_heads

        # 为每个 head / C 创建一个独立的 MCC metric
        self.mcc_metrics = torch.nn.ModuleList([
            MulticlassMatthewsCorrCoef(
                num_classes=num_classes
            )
            for _ in range(num_heads)
        ])

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:  (N, C, K)
        target: (N, C)
        """
        pred_flat = preds.detach().reshape(-1, self.num_heads, 2)  # (N, num_tracks)
        target_flat = targets.detach().reshape(-1, self.num_heads) 
        if pred_flat.ndim != 3:
            raise ValueError(f"preds must be (N, C, K), got {preds.shape}")
        if target_flat.ndim != 2:
            raise ValueError(f"target must be (N, C), got {target_flat.shape}")

        N, C, K = pred_flat.shape
        assert C == self.num_heads, "C dimension mismatch"
        assert K == self.num_classes, "num_classes mismatch"

        for c in range(self.num_heads):
            self.mcc_metrics[c].update(
                pred_flat[:, c, :],    # (N, K)
                target_flat[:, c]       # (N,)
            )

    def compute(self):
        # 每个 C 一个 MCC
        mccs = torch.stack([
            metric.compute()
            for metric in self.mcc_metrics
        ])  # (C,)
        return mccs

    def reset(self):
        for metric in self.mcc_metrics:
            metric.reset()
class BinaryAccuracyWithLogits(BinaryAccuracy):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        super().update(preds, targets)

class BinaryPrecisionWithLogits(BinaryPrecision):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        super().update(preds, targets)  
        
class BinaryRecallWithLogits(BinaryRecall):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        super().update(preds, targets)  
        
class BinaryF1ScoreWithLogits(BinaryF1Score):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        super().update(preds, targets)  
        
class BinaryAUROCeWithLogits(BinaryAUROC):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits,dim=1)
        pos_probs = probs[:, 1]
        super().update(pos_probs, targets)

class BinaryAveragePrecisionWithLogits(BinaryAveragePrecision):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits,dim=1)
        pos_probs = probs[:, 1]
        super().update(pos_probs, targets)   

class BinaryPRAUCWithLogits(BinaryPrecisionRecallCurve):
    """Trapezoidal PR-AUC, matching sklearn auc(recall, precision)."""

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        super().update(probs[:, 1], targets)

    def compute(self):
        precision, recall, _ = super().compute()
        return -torch.trapezoid(precision, recall)
        
class BinaryMatthewsCorrCoefWithLogits(BinaryMatthewsCorrCoef):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        super().update(preds, targets) 
        
def get_metrics(kwargs):
    metrics = nn.ModuleList()

    requested = set(kwargs["types"])
    matched = set()

    if "MCC" in requested:
        metrics.append(MultiHeadMulticlassMCC(2, len(kwargs["class_names"])))
        matched.add("MCC")
    if "PCC" in requested:
        metrics.append(GlobalPearsonCorrCoef(len(kwargs["class_names"])))
        matched.add("PCC")
    if "ACC" in requested:
        metrics.append(BinaryAccuracyWithLogits())
        matched.add("ACC")
    if "Precision" in requested:
        metrics.append(BinaryPrecisionWithLogits())
        matched.add("Precision")
    if "Recall" in requested:
        metrics.append(BinaryRecallWithLogits())
        matched.add("Recall")
    if "F1" in requested:
        metrics.append(BinaryF1ScoreWithLogits())
        matched.add("F1")
    if "AUC" in requested:
        metrics.append(BinaryAUROCeWithLogits())
        matched.add("AUC")
    if "AP" in requested:
        metrics.append(BinaryAveragePrecisionWithLogits())
        matched.add("AP")
    if "AUPRC" in requested:
        metrics.append(BinaryPRAUCWithLogits())
        matched.add("AUPRC")
    if "B_MCC" in requested:
        metrics.append(BinaryMatthewsCorrCoefWithLogits())
        matched.add("B_MCC")
    #  打印未匹配到的 types
    unmatched = requested - matched
    if unmatched:
        print(f"[Warning] Unrecognized metric types: {sorted(unmatched)}")
    else:
        print("Recognized all metric types: {}".format(matched))
    return metrics
