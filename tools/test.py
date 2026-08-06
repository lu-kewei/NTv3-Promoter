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
    metrics_mean_str = ""
    result = {"epoch": int(epoch), "loss": log_sum_loss / (iter + 1)}
    
    for m_iter,_metrics in enumerate(metrics):
        m_values = _metrics.compute().cpu()
        if m_values.dim() == 0:
            m_values = m_values.unsqueeze(0)
        m_values = m_values.numpy()  
        metrics_mean = m_values.mean()
        result[metrics_type[m_iter]] = float(metrics_mean)
        metrics_mean_str = metrics_mean_str+", mean_{}: {:.5f}".format(metrics_type[m_iter],metrics_mean)
        for head_iter, name in enumerate(head_names):
            logger.info("{}/{}: {:.4f}".format(name, metrics_type[m_iter], m_values[head_iter]))
        _metrics.reset()
    logger.info("\n")
    logger.info("epoch: {}, step: {}, loss: {:.5f}, {}".format(epoch, step, log_sum_loss/(iter+1), metrics_mean_str))
    logger.info("\n")
    return result
        

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

    model = instantiate_from_config(cfg.model)

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
            str(model)+'\n'+
            dash_line)
    
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if cfg.ckpt_path is not None:
        logger.info("Load pre_train model...")
        resume_dict = torch.load(cfg.ckpt_path)
        if "trainable_state_dict" in resume_dict:
            model.load_state_dict(resume_dict["trainable_state_dict"], strict=False)
        else:
            load_checkpoints(model,resume_dict)
    else:            
        logger.info("No pre_train model")
        
    metrics_type = cfg.metrics["types"]
    criterion = Criterion(cfg.loss)
    class_names = cfg.class_names

    cfg.data.batch_size = 64
    test_dataset = instantiate_from_config(cfg.data.test_data)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size = cfg.data.batch_size,
        shuffle = False,
        num_workers = cfg.data.workers_per_gpu,
    )
    test_metrics = get_metrics(cfg.metrics)
    model, test_data_loader,test_metrics = accelerator.prepare(model,test_data_loader,test_metrics)
    test(accelerator,model,logger,0,0,test_data_loader,test_metrics,class_names,criterion, metrics_type)
    
if __name__ == '__main__':
    main()
