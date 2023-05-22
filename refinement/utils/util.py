import os
import time

import torch
import torch.nn as nn 
import torch.nn.functional as F

import logging
from pathlib import Path
import shutil
import numpy as np
import yaml

logger = logging.getLogger(__name__)

def create_logger(arg, phase='train'):
    root_output_dir = Path(arg.output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = arg.dataset
    model = arg.model
    n = 0
    if (root_output_dir / dataset).exists():
        n = len(os.listdir(root_output_dir / dataset))
    exp_name = "{:03d}_{}".format(n, arg.description) 
    final_output_dir = root_output_dir / dataset / exp_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_output_dir = final_output_dir / "logs"
    print('=> creating {}'.format(log_output_dir))
    log_output_dir.mkdir(parents=True, exist_ok=True)

    # logger file and tensorboard dir
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format("train", time_str, phase)
    final_log_file = log_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # copy code
    def copytree(src, dst, symlinks=False, ignore=False):
        dir_filter=["__pycache__", "data", "logs"]
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                if item not in dir_filter:
                    if not os.path.exists(d):
                        os.mkdir(d)
                    copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    code_output_dir = final_output_dir / "code"
    print('=> creating {}'.format(code_output_dir))
    code_output_dir.mkdir(parents=True, exist_ok=True)
    copytree(".", code_output_dir)

    ckpt_output_dir = final_output_dir / "checkpoints"
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(ckpt_output_dir), str(log_output_dir)


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


