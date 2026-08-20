import os
import os.path as osp
import importlib
from omegaconf import OmegaConf
base_key = "_BASE_"

def load_config(config_path):

    config_dir = osp.dirname(config_path)
    base_config = OmegaConf.load(config_path) # 将 YAML 内容转换成 OmegaConf 配置对象(可以按字段名称访问)
    yaml_files = base_config._BASE_ # 获取基础配置
    base_config._BASE_ = None # 清空基础配置字段

    merged_config = base_config # 当前实验的其它配置

    for file in yaml_files:
        file_path = osp.join(config_dir,file)
        if os.path.exists(file_path):
            config = OmegaConf.load(file_path) # OmegaConf把配置文件转换为允许用点号访问的python字典
            merged_config = OmegaConf.merge(config,merged_config) # 合并基础配置和当前实验配置
        else:
            print(f"文件 {file_path} 不存在，跳过合并。")

    return merged_config 

def get_obj_from_str(string):
    module,cls = string.rsplit(".",1)
    return getattr(importlib.import_module(module,package=None),cls)

def instantiate_from_config(config,other_params=None):
    if "target" not in config:
        raise KeyError("Expected key target to instantiate.")
    if other_params is not None:
        return get_obj_from_str(config["target"])(**config.get("params"),**other_params)
    else:
        return get_obj_from_str(config["target"])(**config.get("params"))
