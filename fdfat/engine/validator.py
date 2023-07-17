import time
from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from fdfat import __version__
from fdfat.utils.logger import LOGGER
from fdfat.engine.base import BaseEngine
from fdfat.data.dataloader import LandmarkDataset
from fdfat.utils.utils import LMK_PART_NAMES, read_file_list, render_lmk, generate_graph

from fdfat.engine.loop.validator import val_loop

class ValEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_database()
        self.load_model()
    
    def load_database(self):

        self.target_data_path = self.cfgs.data.val
        if self.cfgs.validation == "train":
            self.target_data_path = self.cfgs.data.train
        elif self.cfgs.validation == "test":
            self.target_data_path = self.cfgs.data.test

        LOGGER.info(f"Load {self.cfgs.validation} data")

        self.dataset = LandmarkDataset(self.cfgs, read_file_list(self.target_data_path, base_path=self.cfgs.data.base_path), aug=False)
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfgs.batch_size, shuffle=False,
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn")
        
        LOGGER.info("Load database DONE")

    def prepare(self):
        self.target_checkpoint_path = self.cfgs.checkpoint if self.cfgs.checkpoint is not None else self.save_best
        checkpoint = torch.load(self.target_checkpoint_path)
        self.load_checkpoint(checkpoint)

        self.loss_fn = getattr(nn, self.cfgs.loss)(reduction='none')

    def do_validate(self, verbose=True):
        start_time = time.time()
        loss_dict = val_loop(self.cfgs, 0, self.dataloader, self.net, self.loss_fn)
        LOGGER.info(f"DONE in {int(time.time() - start_time)}s")

        if verbose:
            print(loss_dict)

        return loss_dict
