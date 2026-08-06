import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__))) # 获取根目录
sys.path.append(BASE_DIR) # 把根目录加入模块搜索路径
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  

import time
import argparse 
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from cell.utils.logger import Logger
from cell.utils.configs import load_config,instantiate_from_config
from cell.utils.utils import load_checkpoints, get_device_info
from accelerate import Accelerator 
from omegaconf import OmegaConf
from cell.utils.optimizer import configure_optimizers
from cell.utils.loss import Criterion
from cell.utils.metrics import get_metrics
from cell.utils.data import dna_collate_fn
from cell.utils.training_efficiency import TrainingEfficiencyTracker
from tools.test import test
from accelerate.utils import DistributedDataParallelKwargs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument("--resume_path",type=str,default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"]
    )
    args = parser.parse_args()
    return args


def seed_everything(seed: int) -> torch.Generator:
    """Seed common RNGs without forcing unsupported deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def atomic_write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        temporary = Path(f.name)
    temporary.replace(path)


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        temporary = Path(f.name)
    temporary.replace(path)


    
def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision=args.mixed_precision)
    cfg = load_config(args.config) # 加载配置文件
    configured_seed = args.seed if args.seed is not None else cfg.get("seed")
    seed = int(configured_seed) if configured_seed is not None else None
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max_epochs must be >= 1")
        cfg.max_epochs = args.max_epochs
    cfg.seed = seed
    data_generator = seed_everything(seed) if seed is not None else None
    if args.work_dir is None: # 如果没有指定工作目录，则根据配置文件名和当前时间生成工作目录
        localtime = time.strftime("%Y_%m_%d_%H_%M")
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0],localtime)
        
    log_dir = osp.join(args.work_dir,"log") # 日志路径
    ckpt_dir = osp.join(args.work_dir,"ckpt") # 检查点路径

    os.makedirs(log_dir,exist_ok = True)
    os.makedirs(ckpt_dir,exist_ok = True)

    hpp_logger = Logger(osp.join(log_dir,"hyperparams.log"),name="hyperparams",show=False) # 超参数日志记录器
    logger = Logger(osp.join(log_dir,"train.log"),name="train") # 训练日志记录器
    
    
    model = instantiate_from_config(cfg.model) # 根据配置创建模型，cfg会读取相应yaml文件的配置
    parameter_counts = {
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
        "pooling": sum(p.numel() for p in model.pool.parameters()),
        "classification_head": sum(p.numel() for p in model.bed_head.parameters()),
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if accelerator.is_main_process:
        atomic_write_json(Path(args.work_dir) / "parameter_counts.json", parameter_counts)
    
    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()]) # 字典转字符串

    hpp_logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line)
    hpp_logger.info('cfg info:\n'
            + dash_line + 
            json.dumps(OmegaConf.to_container(cfg), indent=4)+'\n'+
            dash_line) 
    hpp_logger.info(f'seed: {seed if seed is not None else "unset"}')
    hpp_logger.info('Model info:\n'
            + dash_line + 
            str(model)+'\n'+
            dash_line)
    
    collate_fn = None

    if "huggingface_data" in cfg.data.keys():
        dataset, labels = instantiate_from_config(cfg.data.huggingface_data)
        train_dataset = dataset["train"]
        label2id = {label: i for i, label in enumerate(labels)}
        collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=model.dna_tokenizer, label2id=label2id, max_length=cfg.max_length_dna)

    else:
        train_dataset = instantiate_from_config(cfg.data.train_data)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size = cfg.data.batch_size,
        shuffle = True,
        collate_fn=collate_fn,
        num_workers = cfg.data.workers_per_gpu,
        generator=data_generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )

    if cfg.val.flag:
        if "huggingface_data" in cfg.data.keys():
            val_dataset = dataset["val"]
        else:
            val_dataset = instantiate_from_config(cfg.data.val_data)
        val_data_loader = DataLoader(
            val_dataset,
            batch_size = cfg.data.batch_size,
            shuffle = False,
            num_workers = cfg.data.workers_per_gpu,
            worker_init_fn=seed_worker if seed is not None else None,
        )
        val_metrics = get_metrics(cfg.metrics)
        val_data_loader, val_metrics = accelerator.prepare(val_data_loader,val_metrics)
    if cfg.test.flag:
        if "huggingface_data" in cfg.data.keys():
            test_dataset = dataset["test"]
        else:
            test_dataset = instantiate_from_config(cfg.data.test_data)
        test_data_loader = DataLoader(
            test_dataset,
            batch_size = cfg.data.batch_size,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = cfg.data.workers_per_gpu,
            worker_init_fn=seed_worker if seed is not None else None,
        )
        test_metrics = get_metrics(cfg.metrics)
        test_data_loader,test_metrics = accelerator.prepare(test_data_loader,test_metrics)


    start_epoch = 0
    resume_optimizer_state_dict = None
    resume_scheduler_state_dict = None
    
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if args.resume_path is not None:
        cfg.resume_path = args.resume_path

    if cfg.resume_path is not None: # 恢复中断的训练
        logger.info("Load resume...")
        resume_dict = torch.load(cfg.resume_path, map_location="cpu")
        start_epoch = resume_dict["epoch"] + 1
        load_checkpoints(model, resume_dict)
        resume_optimizer_state_dict = resume_dict.get("optim_state_dict")
        resume_scheduler_state_dict = resume_dict.get("scheduler_state_dict")
    elif cfg.ckpt_path is not None:
        logger.info("Load pre_train model...")
        resume_dict = torch.load(cfg.ckpt_path, map_location="cpu")
        if "trainable_state_dict" in resume_dict:
            model.load_state_dict(resume_dict["trainable_state_dict"], strict=False)
        else:
            load_checkpoints(model,resume_dict)
    else:            
        logger.info("No pre_train model")
    
    # from modelscope import AutoModel
    # model.segmentnt = AutoModel.from_pretrained(cfg.segmentnt_name, trust_remote_code=True).to("cuda")
    model, train_data_loader = accelerator.prepare(model,train_data_loader)
    total_steps = len(train_data_loader) * cfg.max_epochs
    optimizer,scheduler = configure_optimizers(model, total_steps, cfg.optimizer) # 创建优化器和学习率调度器（学习率调度器用于在训练过程中改变学习率）
    if resume_optimizer_state_dict is not None:
        optimizer.load_state_dict(resume_optimizer_state_dict)
    if scheduler is not None and resume_scheduler_state_dict is not None:
        scheduler.load_state_dict(resume_scheduler_state_dict)
    if scheduler is not None:
        optimizer,scheduler = accelerator.prepare(optimizer,scheduler)
    else:
        optimizer = accelerator.prepare(optimizer)

    criterion = Criterion(cfg.loss)
    if cfg.metrics.train_flag:
        train_metrics = get_metrics(cfg.metrics)
        train_metrics = accelerator.prepare(train_metrics)
    metrics_type = cfg.metrics["types"]
    class_names = cfg.class_names
    iter_num = len(train_data_loader)  # 一个epoch的迭代数（样本数/batch_size）
    efficiency_tracker = TrainingEfficiencyTracker(
        accelerator=accelerator,
        work_dir=args.work_dir,
        logger=logger,
        batch_size=cfg.data.batch_size,
        resume_existing=start_epoch > 0,
    )
    metrics_path = Path(args.work_dir) / "epoch_metrics.jsonl"
    epoch_metric_records = []
    if start_epoch > 0 and metrics_path.exists():
        epoch_metric_records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    for epoch in range(start_epoch,cfg.max_epochs):
        epoch_loss = 0
        start_time = time.time()
        log_sum_loss = 0
        model = model.train()
        efficiency_epoch = epoch + 1
        track_efficiency = efficiency_tracker.start_epoch(efficiency_epoch)
        for iter, batch in enumerate(train_data_loader):
            iter += 1
            targets = batch["targets"]
            efficiency_tracker.record_batch(targets)
            with accelerator.autocast():
                outputs = model(batch)
                
            loss = criterion(outputs,targets)
            if cfg.metrics.train_flag:
                for _metrics in train_metrics:
                    _metrics.update(
                        outputs,
                        targets
                    )
            # loss = criterion(outs, labels)
            epoch_loss += loss.item()
            log_sum_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            if iter % cfg.log_config.interval == 0:
                if track_efficiency:
                    efficiency_tracker.pause_for_excluded_work()
                try:
                    lr = optimizer.state_dict()["param_groups"][0]["lr"]
                    iter_len = len(str(iter_num))
                    if cfg.metrics.train_flag:
                        metrics_values = train_metrics[0].compute().cpu()
                        if metrics_values.dim() == 0:
                            metrics_values = metrics_values.unsqueeze(0)
                        metrics_values = metrics_values.numpy()
                        metrics_mean = metrics_values.mean()
                        logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}, {}: {:.5f}".format(epoch,iter,iter_len,iter_num,lr,log_sum_loss/(cfg.log_config.interval),metrics_type[0],metrics_mean))
                        for _metrics in train_metrics:
                            _metrics.reset()
                    else:
                        logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}".format(epoch,iter,iter_len,iter_num,lr,log_sum_loss/(cfg.log_config.interval)))
                    log_sum_loss = 0
                finally:
                    if track_efficiency:
                        efficiency_tracker.resume_after_excluded_work()
            # break
            if cfg.val.flag and iter % cfg.val.interval == 0:
                if track_efficiency:
                    efficiency_tracker.pause_for_excluded_work()
                try:
                    test(accelerator,model,logger,epoch, iter,val_data_loader,val_metrics,class_names, criterion, metrics_type, eval_flag=True)
                    model = model.train()
                finally:
                    if track_efficiency:
                        efficiency_tracker.resume_after_excluded_work()
        end_time = time.time()
        efficiency_tracker.finish_epoch(efficiency_epoch)
        logger.info("epoch: {}, Mean Loss: {:.5f}, time: {:.2f} s\n".format(epoch,epoch_loss/iter_num,end_time-start_time))

        if epoch % cfg.checkpoint_config.interval == 0:
            model_state = accelerator.unwrap_model(model).state_dict()
            checkpoint_dict = {
                "epoch": epoch, 
                "state_dict": model_state, 
                "optim_state_dict": optimizer.state_dict(), 
            }
            if scheduler is not None:
                checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
            accelerator.save(checkpoint_dict,osp.join(ckpt_dir,"epoch_"+str(epoch)+".pth")) 
        

        if cfg.test.flag and epoch % cfg.test.epoch_interval == 0:
            epoch_result = test(accelerator,model,logger,epoch,iter,test_data_loader,test_metrics,class_names,criterion, metrics_type)
            epoch_result.update({"seed": seed, "config": str(args.config)})
            epoch_metric_records.append(epoch_result)
            if accelerator.is_main_process:
                atomic_write_jsonl(metrics_path, epoch_metric_records)

    efficiency_tracker.log_summary()
    accelerator.end_training()
            
if __name__ == '__main__':
    main()
