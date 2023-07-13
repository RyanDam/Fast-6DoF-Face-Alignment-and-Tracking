import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union
from datetime import datetime, timedelta
from copy import deepcopy
from PIL import Image
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ConstantLR, SequentialLR, LinearLR, ReduceLROnPlateau

from . import __version__

from fdfat.nn import model
from fdfat.utils.logger import LOGGER
from fdfat.data.dataloader import LandmarkDataset
from fdfat.utils.utils import LMK_PART_NAMES, increment_path, generate_graph, read_file_list, render_lmk
from fdfat.utils.model_utils import init_seeds, select_device, model_info, preprocess
from fdfat.cfg import get_cfg, get_cfg_data, yaml_save, cfg2dict, yaml_print

from fdfat.engine.trainer import train_loop
from fdfat.engine.validator import val_loop

class BaseEngine:

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):

        if isinstance(cfg_, (str, Path)):
            self.cfgs = get_cfg(cfg_)  # load dict
        elif isinstance(cfg_, (Dict)):
            self.cfgs = SimpleNamespace(**vars(cfg_))
        else:
            self.cfgs = cfg_

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

    def load_database(self):
        raise NotImplementedError("Not implemented")
    
    def prepare(self):
        raise NotImplementedError("Not implemented")

    def load_model(self, verbose=False):
        LOGGER.info(f"Load model: {self.cfgs.model}")
        self.net = getattr(model, self.cfgs.model)(imgz=self.cfgs.imgsz, muliplier=self.cfgs.muliplier, pose_rotation=self.cfgs.aux_pose).to(self.cfgs.device)
        if verbose:
            _ = model_info(self.net, detailed=True, imgsz=self.cfgs.imgsz, device=self.cfgs.device)

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            checkpoint = torch.load(checkpoint)
        self.checkpoint = checkpoint
        self.net.load(checkpoint)
        self.start_epoch = checkpoint['epoch']
        self.best_epoch_loss = checkpoint['best_fit']
        self.best_epoch_no = checkpoint['best_epoch']
        LOGGER.info(f"Loaded checkpoint epoch {checkpoint['epoch']}")

class TestEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_model()

        self.target_checkpoint_path = self.cfgs.checkpoint if self.cfgs.checkpoint is not None else self.save_best
        self.load_checkpoint(self.target_checkpoint_path)

        self.net.eval()

    def predict(self, input, render=False):

        if isinstance(input, str):
            input = Image.open(input)

        preprocessed = preprocess(input, self.cfgs.imgsz)
        preprocessed = torch.from_numpy(preprocessed.astype(np.float32)).to(self.cfgs.device)
        with torch.no_grad():
            if self.cfgs.warmup:
                for _ in range(5):
                    _ = self.net(preprocessed)
            start = time.time()
            y = self.net(preprocessed)
            y = y.detach().cpu().numpy()
            end = time.time()

        LOGGER.info(f"Predicted in {int((end-start)*1000):d}ms")

        lmk = y[0][:70*2].reshape((70,2))

        if render:
            rendered = render_lmk(input.copy(), (lmk+0.5)*input.size[0], point_size=1)
            return lmk, rendered

        return lmk

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

        self.dataset = LandmarkDataset(self.cfgs, read_file_list(self.target_data_path, base_path=self.cfgs.data.base_path), 
                                        imgsz=self.cfgs.imgsz, 
                                        pose_rotation=self.cfgs.aux_pose, 
                                        aug=False)
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

class TrainEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_database()
        self.load_model()

    def load_database(self):
        LOGGER.info("Load Train data")
        self.dataset = LandmarkDataset(self.cfgs, read_file_list(self.cfgs.data.train, base_path=self.cfgs.data.base_path), 
                                    imgsz=self.cfgs.imgsz, 
                                    pose_rotation=self.cfgs.aux_pose)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.cfgs.batch_size, shuffle=True, 
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn")

        LOGGER.info("Load Val data")
        self.dataset_test = LandmarkDataset(self.cfgs, read_file_list(self.cfgs.data.val, base_path=self.cfgs.data.base_path), 
                                        imgsz=self.cfgs.imgsz, 
                                        pose_rotation=self.cfgs.aux_pose, 
                                        aug=False)
        self.test_dataloader = DataLoader(self.dataset_test, batch_size=self.cfgs.batch_size, shuffle=False,
                                        pin_memory=self.cfgs.pin_memory,
                                        num_workers=self.cfgs.workers,
                                        persistent_workers=True,
                                        multiprocessing_context="spawn")
        

        LOGGER.info("Load database DONE")

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer']) 

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

    def prepare(self):
        self.loss_fn = getattr(nn, self.cfgs.loss)(reduction='none')
        if self.cfgs.optimizer == "SGD":
            self.optimizer = getattr(torch.optim, self.cfgs.optimizer)(self.net.parameters(), lr=self.cfgs.lr, momentum=0.95, nesterov=True)
        else:
            self.optimizer = getattr(torch.optim, self.cfgs.optimizer)(self.net.parameters(), lr=self.cfgs.lr)

        if self.cfgs.resume:
            self.load_checkpoint(self.save_last)

        scheduler_warmup = ConstantLR(self.optimizer, factor=self.cfgs.lr0_factor, total_iters=self.cfgs.warmup)
        scheduler_main = LinearLR(self.optimizer, start_factor=1, end_factor=0.1, total_iters=self.cfgs.epoch-self.cfgs.warmup)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[self.cfgs.warmup], last_epoch=self.start_epoch)

        if not self.cfgs.resume:
            if self.cfgs.save:
                with open(self.save_log_csv, "a") as f:
                    fields = '\t'.join(LMK_PART_NAMES)
                    fields_test = '\t'.join([f"test_{a}" for a in LMK_PART_NAMES])
                    f.write(f"epoch\ttotal\tnme\t{fields}\tpose\ttest_total\ttest_nme\t{fields_test}\ttest_pose\n")

    def do_train(self):
        start_train_time = time.time()
        for current_epoch in range(self.start_epoch, self.cfgs.epoch):
            LOGGER.info(f"\n\nEPOCH {current_epoch+1}, lr: {self.scheduler.get_last_lr()[0]:>7f}")
            
            train_loss_dict = train_loop(self.cfgs, current_epoch, self.train_dataloader, self.net, self.loss_fn, self.optimizer)
            test_loss_dict = val_loop(self.cfgs, current_epoch, self.test_dataloader, self.net, self.loss_fn)
            self.scheduler.step()

            if test_loss_dict["total"] < best_epoch_loss:
                best_epoch_loss = test_loss_dict["total"]
                best_epoch_no = current_epoch
            else:
                if current_epoch - best_epoch_no > self.cfgs.patience:
                    LOGGER.info(f"STOPPED due to no improvement after {current_epoch - best_epoch_no} epochs")
                    break
            
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
                if best_epoch_no == current_epoch:
                    self.save_model(current_epoch, self.save_best)
                    LOGGER.info(f"---> Saved best: epoch: {current_epoch+1}, loss: {best_epoch_loss:>7f}")

        LOGGER.info(f"DONE in {int(time.time() - start_train_time)}s, best epoch: {best_epoch_no}, val loss: {best_epoch_loss:>7f}")

# def do_train(cfg_: Union[str, Path, Dict, SimpleNamespace]):
    
#     if isinstance(cfg_, (str, Path)):
#         cfgs = get_cfg(cfg_)  # load dict
#     elif isinstance(cfg_, (Dict)):
#         cfgs = SimpleNamespace(**vars(cfg_))
#     else:
#         cfgs = cfg_

#     init_seeds(cfgs.seed)

#     save_dir = Path(increment_path(Path(cfgs.project) / cfgs.name, mkdir=not cfgs.resume))
#     LOGGER.info(f"Project path: {save_dir}")
#     save_wdir = save_dir / 'weights'  # weights dir
#     save_wdir.mkdir(parents=True, exist_ok=True)  # make directory
#     save_last, save_best = save_wdir / 'last.pt', save_wdir / 'best.pt'  # checkpoint paths
#     save_log_csv = save_dir / 'log.csv'
#     save_log_png = save_dir / 'log.png'
#     save_config = save_dir / 'config.yaml'
#     yaml_save(save_config, cfg2dict(cfgs))
#     yaml_print(cfgs)

#     cfgs.save_dir = save_dir
#     if cfgs.data is not None:
#         cfgs.data = get_cfg_data(cfgs.data)

#     cfgs.device = select_device(cfgs.device)
    
#     LOGGER.info(f"Using {cfgs.device} device")

#     LOGGER.info("Load Train data")
#     dataset = LandmarkDataset(cfgs, read_file_list(cfgs.data.train, base_path=cfgs.data.base_path), 
#                                 imgsz=cfgs.imgsz, 
#                                 pose_rotation=cfgs.aux_pose)
#     train_dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, 
#                                     pin_memory=cfgs.pin_memory,
#                                     num_workers=cfgs.workers,
#                                     persistent_workers=True,
#                                     multiprocessing_context="spawn")

#     LOGGER.info("Load Val data")
#     dataset_test = LandmarkDataset(cfgs, read_file_list(cfgs.data.val, base_path=cfgs.data.base_path), 
#                                     imgsz=cfgs.imgsz, 
#                                     pose_rotation=cfgs.aux_pose, 
#                                     aug=False)
#     test_dataloader = DataLoader(dataset_test, batch_size=cfgs.batch_size, shuffle=False,
#                                     pin_memory=cfgs.pin_memory,
#                                     num_workers=cfgs.workers,
#                                     persistent_workers=True,
#                                     multiprocessing_context="spawn")
#     start_epoch = 0
#     best_epoch_loss = 999
#     best_epoch_no = 0

#     LOGGER.info(f"Load Model: {cfgs.model}")
#     net = getattr(model, cfgs.model)(imgz=cfgs.imgsz, muliplier=cfgs.muliplier, pose_rotation=cfgs.aux_pose).to(cfgs.device)

#     loss_fn = getattr(nn, cfgs.loss)(reduction='none')
#     if cfgs.optimizer == "SGD":
#         optimizer = getattr(torch.optim, cfgs.optimizer)(net.parameters(), lr=cfgs.lr, momentum=0.95, nesterov=True)
#     else:
#         optimizer = getattr(torch.optim, cfgs.optimizer)(net.parameters(), lr=cfgs.lr)

#     if cfgs.resume:
#         checkpoint = torch.load(save_best)
#         net.load(checkpoint)

#         start_epoch = checkpoint['epoch']
#         best_epoch_loss = checkpoint['best_fit']
#         best_epoch_no = checkpoint['best_epoch']

#     _ = model_info(net, detailed=True, imgsz=cfgs.imgsz, device=cfgs.device)

#     if cfgs.resume:
#         optimizer.load_state_dict(checkpoint['optimizer']) 

#     scheduler_warmup = ConstantLR(optimizer, factor=cfgs.lr0_factor, total_iters=cfgs.warmup)
#     scheduler_main = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=cfgs.epoch-cfgs.warmup)
#     scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[cfgs.warmup], last_epoch=start_epoch)

#     if not cfgs.resume:
#         with open(save_log_csv, "a") as f:
#             fields = '\t'.join(LMK_PART_NAMES)
#             fields_test = '\t'.join([f"test_{a}" for a in LMK_PART_NAMES])
#             f.write(f"epoch\ttotal\tnme\t{fields}\tpose\ttest_total\ttest_nme\t{fields_test}\ttest_pose\n")

#     def save_model(epoch, save_path):
#         ckpt = {
#             'epoch': epoch,
#             'model': deepcopy(net).half(),
#             'best_fit': best_epoch_loss,
#             'best_epoch': best_epoch_no,
#             'optimizer': optimizer.state_dict(),
#             'train_args': cfgs,
#             'date': datetime.now().isoformat(),
#             'version': __version__
#         }
#         torch.save(ckpt, save_path)

#     start_train_time = time.time()
#     for current_epoch in range(start_epoch, cfgs.epoch):
#         LOGGER.info(f"\n\nEPOCH {current_epoch+1}, lr: {scheduler.get_last_lr()[0]:>7f}")
        
#         train_loss_dict = train_loop(cfgs, current_epoch, train_dataloader, net, loss_fn, optimizer)
#         test_loss_dict = val_loop(cfgs, current_epoch, test_dataloader, net, loss_fn)
#         scheduler.step()

#         with open(save_log_csv, "a") as f:
#             f.write(f"{current_epoch+1}")
#             f.write(f"\t{train_loss_dict['total']}")
#             f.write(f"\t{train_loss_dict['nme']}")
#             for n in LMK_PART_NAMES:
#                 f.write(f"\t{train_loss_dict[n]}")
#             f.write(f"\t{train_loss_dict['pose']}")

#             f.write(f"\t{test_loss_dict['total']}")
#             f.write(f"\t{test_loss_dict['nme']}")
#             for n in LMK_PART_NAMES:
#                 f.write(f"\t{test_loss_dict[n]}")
#             f.write(f"\t{test_loss_dict['pose']}")
            
#             f.write(f"\n")

#         generate_graph(save_log_csv, save_log_png)

#         save_model(current_epoch, save_last)
#         if test_loss_dict["total"] < best_epoch_loss:
#             best_epoch_loss = test_loss_dict["total"]
#             best_epoch_no = current_epoch
#             save_model(current_epoch, save_best)
#             LOGGER.info(f"---> Saved best: epoch: {current_epoch+1}, loss: {best_epoch_loss:>7f}")
#         else:
#             if current_epoch - best_epoch_no > cfgs.patience:
#                 LOGGER.info(f"STOPPED due to no improvement after {current_epoch - best_epoch_no} epochs")
#                 break
            
#     LOGGER.info(f"DONE in {int(time.time() - start_train_time)}s, best epoch: {best_epoch_no}, val loss: {best_epoch_loss:>7f}")

