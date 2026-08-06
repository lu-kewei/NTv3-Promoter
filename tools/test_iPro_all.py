import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from cell.utils.logger import Logger
from cell.utils.configs import load_config, instantiate_from_config
from cell.utils.utils import load_checkpoints, get_device_info
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator 
import argparse 
import json 
from omegaconf import OmegaConf
import time 
from tqdm import tqdm 
from cell.utils.metrics import get_metrics
from cell.utils.loss import Criterion
import numpy as np 
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"]
    )
    args = parser.parse_args()
    return args


def test(accelerator,model,logger,epoch,step,data_loader,metrics,head_names,criterion, metrics_type,eval_flag=False):
    model.eval()
    log_sum_loss = 0 
    logger.info("\n")
    if eval_flag:
        logger.info("Eval...")
    else:
        logger.info("Test...")
    for iter, batch in enumerate(tqdm(data_loader)):
        targets = batch["targets"]
        with torch.no_grad():
            with accelerator.autocast():
                outputs = model(batch)

        # Compute loss
        loss = criterion(
            outputs,
            targets,
        )
        for _metrics in metrics:
            _metrics.update(
                outputs,
                targets
            )
        log_sum_loss += loss.item()
    metrics_mean_str = " "
    
    for m_iter,_metrics in enumerate(metrics):
        m_values = _metrics.compute().cpu()
        if m_values.dim() == 0:
            m_values = m_values.unsqueeze(0)
        m_values = m_values.numpy()  
        metrics_mean = m_values.mean()
        if metrics_mean_str==" ":
            metrics_mean_str = ", mean_{}: {:.5f}".format(metrics_type[m_iter],metrics_mean)
        else:
            metrics_mean_str = metrics_mean_str+", mean_{}: {:.5f}".format(metrics_type[m_iter],metrics_mean)
        for head_iter, name in enumerate(head_names):
            logger.info("{}/{}: {:.4f}".format(name, metrics_type[m_iter], m_values[head_iter]))
        _metrics.reset()
    logger.info("\n")
    logger.info("epoch: {}, step: {}, loss: {:.5f} {}".format(epoch,step,log_sum_loss/(iter+1),metrics_mean_str))
    logger.info("\n")
    return metrics_mean_str  

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    cfg = load_config(args.config)

    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])
    test_dir = osp.join(args.work_dir,"test")
    log_dir = osp.join(test_dir,"log")

    os.makedirs(log_dir,exist_ok=True)

    localtime = time.strftime("%Y_%m_%d_%H_%M")
    logger = Logger(osp.join(log_dir,"{}.log".format(localtime)),name="test")

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])

    _model = instantiate_from_config(cfg.model)

    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line)
    logger.info('cfg info:\n'
            + dash_line + 
            json.dumps(OmegaConf.to_container(cfg), indent=4)+'\n'+
            dash_line) 
    logger.info('Model info:\n'
            + dash_line + 
            str(_model)+'\n'+
            dash_line)
    
    if args.ckpt_path is None:
        raise ValueError("Please specify a joint-model checkpoint with --ckpt_path")
    if not osp.isfile(args.ckpt_path):
        raise FileNotFoundError("Checkpoint does not exist: {}".format(args.ckpt_path))

    logger.info("Load joint-model checkpoint: {}...".format(args.ckpt_path))
    resume_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "trainable_state_dict" in resume_dict:
        _model.load_state_dict(resume_dict["trainable_state_dict"], strict=False)
    else:
        load_checkpoints(_model, resume_dict)
    checkpoint_epoch = resume_dict.get("epoch", 0)
    model = accelerator.prepare(_model)
        
    metrics_type = cfg.metrics["types"]
    criterion = Criterion(cfg.loss)
    class_names = cfg.class_names

    data_path = cfg.data.test_data.params.data_path
    data_dir = osp.dirname(data_path)
    species_ID = ["Acinetobacter baumannii ATCC 17978", # 1
            "Bradyrhizobium japonicum USDA 110",   # 2
            "Burkholderia cenocepacia J2315", # 3
            "Campylobacter jejuni RM1221", # 4
            "Campylobacter jejuni subsp. jejuni 81116", # 5
            "Campylobacter jejuni subsp. jejuni 81-176", # 6
            "Campylobacter jejuni subsp. jejuni NCTC 11168", # 7
            "Corynebacterium diphtheriae NCTC 13129", # 8
            "Corynebacterium glutamicum ATCC 13032", # 9
            "Escherichia coli str K-12 substr. MG1655", # 10
            "Haloferax volcanii DS2", # 11
            "Helicobacter pylori strain 26695", # 12
            "Nostoc sp. PCC7120",  # 13
            "Paenibacillus riograndensis SBR5", # 14
            "Pseudomonas putida KT2440",  # 15
            "Shigella flexneri 5a str. M90T", # 16
            "Sinorhizobium meliloti 1021", # 17
            "Staphylococcus aureus subsp. aureus MW2", # 18
            "Staphylococcus epidermidis ATCC 12228", # 19
            "Synechococcus elongatus PCC 7942", # 20
            "Thermococcus kodakarensis KOD1", # 21
            "Xanthomonas campestris pv. campestrie B100",  # 22
            "Bacillus subtilis subsp. subtilis str. 168"   #23
            ]
    metrics_mean_list = []
    for num, data_name in enumerate(species_ID):
        data_name = data_name.replace(" ","_")
        logger.info("Num: {}, {}".format(num+1,data_name))
        data_path = osp.join(data_dir,data_name+".csv")
        if not osp.exists(data_path):
            raise Exception("{} is not exist!".format(data_path))
        cfg.data.test_data.params.data_path = data_path
        test_dataset = instantiate_from_config(cfg.data.test_data)
        test_data_loader = DataLoader(
            test_dataset,
            batch_size = cfg.data.batch_size,
            shuffle = False,
            num_workers = cfg.data.workers_per_gpu,
        )
        test_metrics = get_metrics(cfg.metrics)
        test_data_loader, test_metrics = accelerator.prepare(
            test_data_loader,
            test_metrics,
        )
        metrics_mean_str = test(
            accelerator,
            model,
            logger,
            checkpoint_epoch,
            0,
            test_data_loader,
            test_metrics,
            class_names,
            criterion,
            metrics_type,
        )
        metrics_mean_list.append(metrics_mean_str)
    values = defaultdict(list)

    for s in metrics_mean_list:
        for item in s.split(","):
            if item == "":
                continue
            k, v = item.split(":")
            values[k.strip()].append(float(v))

    mean_results = {k: np.mean(v) for k, v in values.items()}
    logger.info(", ".join(f"All_{k}: {v:.5f}" for k, v in mean_results.items()))
if __name__ == '__main__':
    main()
