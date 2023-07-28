import time
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, Union
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ConstantLR, SequentialLR, LinearLR, ReduceLROnPlateau, MultiStepLR

from fdfat import __version__, MACOS, LINUX, WINDOWS
from fdfat.utils.logger import LOGGER
from fdfat.engine.base import BaseEngine
from fdfat.data.dataloader import LandmarkDataset
from fdfat.utils.utils import LMK_PART_NAMES, read_file_list, render_lmk, generate_graph

from fdfat.engine.loop.trainer import train_loop
from fdfat.engine.loop.validator import val_loop

class TrainEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_database()
        self.load_model()

    def load_database(self):
        LOGGER.info("Load Train data")
        self.dataset = LandmarkDataset(self.cfgs, read_file_list(self.cfgs.data.train, base_path=self.cfgs.data.base_path), cache_path=self.cfgs.data.train_cache)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.cfgs.batch_size, shuffle=True, 
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn" if MACOS else None)

        LOGGER.info("Load Val data")
        self.dataset_test = LandmarkDataset(self.cfgs, read_file_list(self.cfgs.data.val, base_path=self.cfgs.data.base_path), cache_path=self.cfgs.data.val_cache, aug=False)
        self.test_dataloader = DataLoader(self.dataset_test, batch_size=self.cfgs.batch_size, shuffle=False,
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn" if MACOS else None)

        LOGGER.info("Load database DONE")

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer']) 

    def save_model(self, epoch, save_path):
        ckpt = {
            'epoch': epoch,
            'model': deepcopy(self.net).half(),
            'best_fit': self.best_epoch_loss,
            'best_epoch': self.best_epoch_no,
            'optimizer': self.optimizer.state_dict(),
            'train_args': self.cfgs,
            'date': datetime.now().isoformat(),
            'version': __version__
        }
        torch.save(ckpt, save_path)

    def current_lr(self) -> [float]:
        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(float(param_group['lr']))
        return lrs

    def prepare(self):
        self.loss_fn = getattr(nn, self.cfgs.loss)(reduction='none')
        if self.cfgs.optimizer == "SGD":
            self.optimizer = getattr(torch.optim, self.cfgs.optimizer)(self.net.parameters(), lr=self.cfgs.lr, momentum=0.95, nesterov=True)
        else:
            self.optimizer = getattr(torch.optim, self.cfgs.optimizer)(self.net.parameters(), lr=self.cfgs.lr)

        if self.cfgs.resume:
            self.load_checkpoint(self.save_last)
        elif self.cfgs.checkpoint is not None:
            super.load_checkpoint(self.cfgs.checkpoint, epoch_info=False) # load weight

        # scheduler_warmup = ConstantLR(self.optimizer, factor=self.cfgs.lr0_factor, total_iters=self.cfgs.warmup_epoch)
        # scheduler_main = LinearLR(self.optimizer, start_factor=1/self.cfgs.lr0_factor, end_factor=self.cfgs.lre_factor, total_iters=self.cfgs.epoch-self.cfgs.warmup_epoch)
        # self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[self.cfgs.warmup_epoch], last_epoch=self.start_epoch)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.cfgs.lre_factor, patience=self.cfgs.patience)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[50,100], gamma=self.cfgs.lre_factor)

        if not self.cfgs.resume:
            if self.cfgs.save:
                with open(self.save_log_csv, "a") as f:
                    fields = '\t'.join(LMK_PART_NAMES)
                    fields_test = '\t'.join([f"test_{a}" for a in LMK_PART_NAMES])
                    f.write(f"epoch\ttotal\tnme\t{fields}\tpose\ttest_total\ttest_nme\t{fields_test}\ttest_pose\n")

    def do_train(self):
        start_train_time = time.time()
        for current_epoch in range(self.start_epoch, self.cfgs.epoch):
            LOGGER.info(f"\n\nEPOCH {current_epoch+1}, lr: {self.current_lr()[0]:0.7f}")
            
            train_loss_dict = train_loop(self.cfgs, current_epoch, self.train_dataloader, self.net, self.loss_fn, self.optimizer)
            test_loss_dict = val_loop(self.cfgs, current_epoch, self.test_dataloader, self.net, self.loss_fn)
            self.scheduler.step()
            # self.scheduler.step(test_loss_dict['total'])

            if test_loss_dict["total"] < self.best_epoch_loss:
                self.best_epoch_loss = test_loss_dict["total"]
                self.best_epoch_no = current_epoch
            # else:
            #     if current_epoch - self.best_epoch_no > self.cfgs.patience:
            #         LOGGER.info(f"STOPPED due to no improvement after {current_epoch - self.best_epoch_no} epochs")
            #         break
            
            if self.cfgs.save:
                with open(self.save_log_csv, "a") as f:
                    f.write(f"{current_epoch+1}")
                    f.write(f"\t{train_loss_dict['total']}")
                    f.write(f"\t{train_loss_dict['nme']}")
                    for n in LMK_PART_NAMES:
                        f.write(f"\t{train_loss_dict[n]}")
                    f.write(f"\t{train_loss_dict['pose']}")

                    f.write(f"\t{test_loss_dict['total']}")
                    f.write(f"\t{test_loss_dict['nme']}")
                    for n in LMK_PART_NAMES:
                        f.write(f"\t{test_loss_dict[n]}")
                    f.write(f"\t{test_loss_dict['pose']}")
                    
                    f.write(f"\n")

                generate_graph(self.save_log_csv, self.save_log_png)

                self.save_model(current_epoch, self.save_last)
                if self.best_epoch_no == current_epoch:
                    self.save_model(current_epoch, self.save_best)
                    LOGGER.info(f"---> Saved best: epoch: {current_epoch+1}, loss: {self.best_epoch_loss:>7f}")

        LOGGER.info(f"DONE in {int(time.time() - start_train_time)}s, best epoch: {self.best_epoch_no}, val loss: {self.best_epoch_loss:>7f}")
