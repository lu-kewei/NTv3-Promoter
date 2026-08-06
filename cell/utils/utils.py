
import torch 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import os.path as osp 
import torch.nn.functional as F 
from Bio import SeqIO

def load_checkpoints(model,resume_dict,strict=False,only_model=False):
    # pretrained_dict = torch.load(checkpoints)
    if "state_dict" not in resume_dict.keys():
        pretrained_dict = resume_dict
    else:
        pretrained_dict = resume_dict["state_dict"]
    if strict is True:
        try: 
            if only_model:
                model.load_state_dict(pretrained_dict)
            else:
                model.model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        if only_model:
            model_dict = model.model.state_dict()
        else:
            model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        not_update_dict = set(model_dict.keys()) - set(pretrained_dict.keys())
        if len(not_update_dict) != 0:
            print("not update params: {}\n".format(not_update_dict))
        model_dict.update(pretrained_dict)
        if only_model:
            model.model.load_state_dict(model_dict,strict=False)
        else:
            model.load_state_dict(model_dict,strict=False)

def get_device_info():
    gpu_info_dict = {}
    if torch.cuda.is_available():
        gpu_info_dict["CUDA available"]=True
        gpu_num = torch.cuda.device_count()
        gpu_info_dict["GPU numbers"]=gpu_num
        infos = [{"GPU "+str(i):torch.cuda.get_device_name(i)} for i in range(gpu_num)]
        gpu_info_dict["GPU INFO"]=infos
    else:
        gpu_info_dict["CUDA_available"]=False
    return gpu_info_dict

import torch
import torch.nn.functional as F

def multiclass_independent_focal_loss(
    logits,          # (B, L, C, 2)
    targets,         # (B, L, C)  in {0,1}
    gamma=2.0,
    alpha=0.25,
    mask=None,       # (B, L) or (B,L,C)  可选
    reduction="mean"
):
    """
    Multi-label binary Focal Loss for (B,L,C,2) logits.
    """

    # softmax -> p_pos
    probs = F.softmax(logits, dim=-1)[..., 1]    # (B,L,C)

    targets = targets.float()

    pt = torch.where(targets == 1, probs, 1 - probs)
    w  = torch.where(targets == 1, alpha, 1 - alpha)
    focal = (1 - pt) ** gamma

    loss = -w * focal * torch.log(pt.clamp(min=1e-8))   # (B,L,C)

    # mask support (for N, padding, etc.)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # (B,L,1)
        loss = loss * mask

    if reduction == "mean":
        return loss.sum() / (mask.sum() if mask is not None else loss.numel())
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def load_genome_fasta(fasta_path,test_flag = False):
    genome = {}
    for rec in SeqIO.parse(fasta_path, "fasta"): 
        chrom = rec.id
        name = chrom[3:] if chrom.startswith("chr") else chrom
        # if name in [str(i) for i in range(1,23+1)] + ["X","Y","MT"]:
        #     genome[name] = str(rec.seq)
        
        if test_flag:
            if name in [str(i) for i in [20,21]]:
                genome[name] = str(rec.seq)
        else:
            # if name in [str(i) for i in range(1,19+1)] + ["23"]+["X","Y","MT"]:
            if name in [str(i) for i in range(1,19+1)] + ["23"]+["X","Y"]:
                genome[name] = str(rec.seq)
            # if name in [str(i) for i in range(11,11+1)]:
            #     genome[name] = str(rec.seq)
    return genome

def plot_auc_auprc(all_probs,all_targets,save_dir,name):
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    # === ROC ===
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(osp.join(save_dir,"{}_AUC.svg".format(name)), dpi=300)
    plt.close()

    # === PR ===
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = average_precision_score(all_targets, all_probs)

    plt.figure()
    plt.plot(recall, precision, label=f'AUPRC = {pr_auc:.4f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(osp.join(save_dir,"{}_AUPRC.svg".format(name)), dpi=300)
    plt.close()
