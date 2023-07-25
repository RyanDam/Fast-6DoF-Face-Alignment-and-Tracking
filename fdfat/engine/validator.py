import time
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from fdfat import __version__, MACOS, LINUX, WINDOWS
from fdfat.utils.logger import LOGGER
from fdfat.engine.base import BaseEngine
from fdfat.data.dataloader import LandmarkDataset
from fdfat.utils.utils import LMK_PART_NAMES, read_file_list, render_lmk, generate_graph, render_lmk_nme

from fdfat.engine.loop.validator import val_loop


class ValEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_database()
        self.load_model()
    
    def load_database(self):

        self.target_data_path = self.cfgs.data.val
        self.target_cache_path = self.cfgs.data.val_cache
        if self.cfgs.validation == "train":
            self.target_data_path = self.cfgs.data.train
            self.target_cache_path = self.cfgs.data.train_cache
        elif self.cfgs.validation == "test":
            self.target_data_path = self.cfgs.data.test
            self.target_cache_path = self.cfgs.data.test_cache
        self.target_data_path = Path(self.target_data_path)
        self.target_cache_path = Path(self.target_cache_path) if self.target_cache_path is not None else None

        LOGGER.info(f"Load {self.cfgs.validation} data")

        self.dataset = LandmarkDataset(self.cfgs, read_file_list(self.target_data_path, base_path=self.cfgs.data.base_path), cache_path=self.target_cache_path, aug=False)
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfgs.batch_size, shuffle=False,
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn" if MACOS else None)
        
        LOGGER.info("Load database DONE")

    def prepare(self):
        self.target_checkpoint_path = self.cfgs.checkpoint if self.cfgs.checkpoint is not None else self.save_best
        self.load_checkpoint(self.target_checkpoint_path)
        self.loss_fn = getattr(nn, self.cfgs.loss)(reduction='none')

    def do_validate(self, verbose=True):
        start_time = time.time()
        loss_dict = val_loop(self.cfgs, 0, self.dataloader, self.net, self.loss_fn)
        LOGGER.info(f"DONE in {int(time.time() - start_time)}s")

        if verbose:
            for k, v in loss_dict.items():
                print(k, v, "\n")

        timenow = datetime.now().isoformat()
        fname = self.target_data_path.stem
        validate_name = f"validate_{self.cfgs.validation}_{fname}_{timenow}"
        validate_record_path = self.save_dir / f"{validate_name}.json"
        validate_lmk_nme_path = self.save_dir / f"{validate_name}.jpg"

        loss_dict["date"] = timenow
        loss_dict["datapath"] = str(self.target_data_path)
        loss_dict["database"] = str(self.cfgs.data.base_path)

        with open(validate_record_path, "w") as f:
            json.dump(loss_dict, f)

        lmk_nme = render_lmk_nme(loss_dict["nme_stat"]["nme_part"]["nme_mean"], title=validate_name)
        lmk_nme.save(validate_lmk_nme_path)
        
        return loss_dict
