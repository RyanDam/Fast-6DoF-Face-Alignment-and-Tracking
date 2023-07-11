import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union
from datetime import datetime, timedelta
from copy import deepcopy
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
from fdfat.utils.utils import LMK_PART_NAMES, increment_path, generate_graph, read_file_list
from fdfat.utils.model_utils import init_seeds, select_device, model_info
from fdfat.cfg import get_cfg, get_cfg_data, yaml_save, cfg2dict, yaml_print

from fdfat.engine.trainer import train_loop
from fdfat.engine.validator import val_loop

def do_train(cfg_: Union[str, Path, Dict, SimpleNamespace]):
    
    if isinstance(cfg_, (str, Path)):
        cfgs = get_cfg(cfg_)  # load dict
    elif isinstance(cfg_, (Dict)):
        cfgs = SimpleNamespace(**vars(cfg_))
    else:
        cfgs = cfg_

    init_seeds(cfgs.seed)

    save_dir = Path(increment_path(Path(cfgs.project) / cfgs.name, mkdir=cfgs.override==False))
    LOGGER.info(f"Project path: {save_dir}")
    save_wdir = save_dir / 'weights'  # weights dir
    save_wdir.mkdir(parents=True, exist_ok=True)  # make directory
    save_last, save_best = save_wdir / 'last.pt', save_wdir / 'best.pt'  # checkpoint paths
    save_log_csv = save_dir / 'log.csv'
    save_log_png = save_dir / 'log.png'
    save_config = save_dir / 'config.yaml'
    yaml_save(save_config, cfg2dict(cfgs))
    yaml_print(cfgs)

    cfgs.save_dir = save_dir
    if cfgs.data is not None:
        cfgs.data = get_cfg_data(cfgs.data)

    cfgs.device = select_device(cfgs.device)
    
    LOGGER.info(f"Using {cfgs.device} device")

    LOGGER.info("Load Train data")
    dataset = LandmarkDataset(cfgs, read_file_list(cfgs.data.train, base_path=cfgs.data.base_path), 
                                imgsz=cfgs.imgsz, 
                                pose_rotation=cfgs.aux_pose)
    train_dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, 
                                    pin_memory=cfgs.pin_memory,
                                    num_workers=cfgs.workers,
                                    persistent_workers=True,
                                    multiprocessing_context="spawn")

    LOGGER.info("Load Val data")
    dataset_test = LandmarkDataset(cfgs, read_file_list(cfgs.data.val, base_path=cfgs.data.base_path), 
                                    imgsz=cfgs.imgsz, 
                                    pose_rotation=cfgs.aux_pose, 
                                    aug=False)
    test_dataloader = DataLoader(dataset_test, batch_size=cfgs.batch_size, shuffle=False,
                                    pin_memory=cfgs.pin_memory,
                                    num_workers=cfgs.workers,
                                    persistent_workers=True,
                                    multiprocessing_context="spawn")
    start_epoch = 0
    LOGGER.info(f"Load Model: {cfgs.model}")
    net = getattr(model, cfgs.model)(imgz=cfgs.imgsz, muliplier=cfgs.muliplier, pose_rotation=cfgs.aux_pose).to(cfgs.device)
    # if cfgs.resume:
    #     checkpoint = torch.load(save_best)
    #     net.load_state_dict(checkpoint)
    #     start_epoch = checkpoint['epoch']

    _ = model_info(net, detailed=True, imgsz=cfgs.imgsz, device=cfgs.device)

    loss_fn = getattr(nn, cfgs.loss)(reduction='none')
    if cfgs.optimizer == "SGD":
        optimizer = getattr(torch.optim, cfgs.optimizer)(net.parameters(), lr=cfgs.lr, momentum=0.95, nesterov=True)
    else:
        optimizer = getattr(torch.optim, cfgs.optimizer)(net.parameters(), lr=cfgs.lr)

    best_epoch_loss = 999
    best_epoch_no = 0
    start_train_time = time.time()

    scheduler_warmup = ConstantLR(optimizer, factor=cfgs.lr0_factor, total_iters=cfgs.warmup)
    scheduler_main = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=cfgs.epoch-cfgs.warmup)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[cfgs.warmup])

    # if not cfgs.resume:
    with open(save_log_csv, "a") as f:
        fields = '\t'.join(LMK_PART_NAMES)
        fields_test = '\t'.join([f"test_{a}" for a in LMK_PART_NAMES])
        f.write(f"epoch\ttotal\tnme\t{fields}\tpose\ttest_total\ttest_nme\t{fields_test}\ttest_pose\n")

    def save_model(epoch, save_path):
        ckpt = {
            'epoch': epoch,
            'model': deepcopy(net).half(),
            'optimizer': optimizer.state_dict(),
            'train_args': cfgs,
            'date': datetime.now().isoformat(),
            'version': __version__
        }
        torch.save(ckpt, save_path)

    for current_epoch in range(start_epoch, cfgs.epoch):
        LOGGER.info(f"\n\nEPOCH {current_epoch+1}, lr: {scheduler.get_last_lr()[0]:>7f}")
        
        train_loss_dict = train_loop(cfgs, current_epoch, train_dataloader, net, loss_fn, optimizer)
        test_loss_dict = val_loop(cfgs, current_epoch, test_dataloader, net, loss_fn)
        scheduler.step()

        with open(save_log_csv, "a") as f:
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

        generate_graph(save_log_csv, save_log_png)

        save_model(current_epoch, save_last)
        # torch.save(net.state_dict(), save_last)
        if test_loss_dict["total"] < best_epoch_loss:
            best_epoch_loss = test_loss_dict["total"]
            best_epoch_no = current_epoch
            # torch.save(net.state_dict(), save_best)
            save_model(current_epoch, save_best)
            LOGGER.info(f"---> Saved best: epoch: {current_epoch+1}, loss: {best_epoch_loss:>7f}")
        else:
            if current_epoch - best_epoch_no > cfgs.patience:
                LOGGER.info(f"STOPPED in {int(time.time() - start_train_time)}s, due to no improvement after {current_epoch - best_epoch_no} epochs, best epoch: {best_epoch_no}, val loss: {best_epoch_loss:>7f}")
                break
            
    LOGGER.info(f"DONE in {int(time.time() - start_train_time)}s, best epoch: {best_epoch_no}, val loss: {best_epoch_loss:>7f}")
