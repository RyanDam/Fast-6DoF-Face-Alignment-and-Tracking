from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace

import torch
from torch import nn

from fdfat.nn import model
from fdfat.utils.logger import LOGGER
from fdfat.cfg import get_cfg, get_cfg_data, yaml_save, cfg2dict, yaml_print
from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES, increment_path, generate_graph, read_file_list, render_lmk
from fdfat.utils.model_utils import init_seeds, select_device, model_info

class BaseEngine:

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):

        if isinstance(cfg_, (str, Path)):
            self.cfgs = get_cfg(cfg_)  # load dict
        elif isinstance(cfg_, (Dict)):
            self.cfgs = SimpleNamespace(**vars(cfg_))
        else:
            self.cfgs = cfg_

        init_seeds(self.cfgs.seed)

        self.save_dir = Path(increment_path(Path(self.cfgs.project) / self.cfgs.name, exist_ok=self.cfgs.task != "train", mkdir=(self.cfgs.task == "train" and not self.cfgs.resume)))
        LOGGER.info(f"Project path: {self.save_dir}")
        self.save_wdir = self.save_dir / 'weights'  # weights dir
        self.save_wdir.mkdir(parents=True, exist_ok=True)  # make directory
        self.save_last, self.save_best = self.save_wdir / 'last.pt', self.save_wdir / 'best.pt'  # checkpoint paths
        self.save_log_csv = self.save_dir / 'log.csv'
        self.save_log_png = self.save_dir / 'log.png'
        self.save_config = self.save_dir / 'config.yaml'
        if self.cfgs.task == "train":
            yaml_save(self.save_config, cfg2dict(self.cfgs))

        yaml_print(self.cfgs)

        self.cfgs.save_dir = self.save_dir
        if self.cfgs.data is not None:
            self.cfgs.data = get_cfg_data(self.cfgs.data)

        self.cfgs.device = select_device(self.cfgs.device)

        LOGGER.info(f"Using {self.cfgs.device} device")

        self.start_epoch = 0
        self.best_epoch_loss = 999
        self.best_epoch_no = 0

        if self.cfgs.lmk_num == 70:
            self.cfgs.lmk_parts = LMK_PARTS
            self.cfgs.lmk_part_names = LMK_PART_NAMES
        else: # 68 points
            self.cfgs.lmk_parts = LMK_PARTS[:-1]
            self.cfgs.lmk_part_names = LMK_PART_NAMES[:-1]

    def load_database(self):
        raise NotImplementedError("Not implemented")
    
    def prepare(self):
        raise NotImplementedError("Not implemented")

    def load_model(self, verbose=True):
        LOGGER.info(f"Load model: {self.cfgs.model}")
        self.net = getattr(model, self.cfgs.model)(imgz=self.cfgs.imgsz, muliplier=self.cfgs.muliplier, pose_rotation=self.cfgs.aux_pose).to(self.cfgs.device)
        if verbose:
            # need to call this before getting model info to prevent abnormal batchnorm
            self.net.eval()
            _ = model_info(self.net, detailed=True, imgsz=self.cfgs.imgsz, device=self.cfgs.device)

    def load_checkpoint(self, checkpoint, map_location='cpu'):
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        self.checkpoint = checkpoint
        self.net.load(checkpoint)
        self.start_epoch = checkpoint['epoch']
        self.best_epoch_loss = checkpoint['best_fit']
        self.best_epoch_no = checkpoint['best_epoch']
        LOGGER.info(f"Loaded checkpoint epoch {checkpoint['epoch']}")