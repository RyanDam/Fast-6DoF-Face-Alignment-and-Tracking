import tqdm
from collections import defaultdict

import torch

from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES, render_batch
from landmark import TQDM_BAR_FORMAT
from fdfat.metric.metric import nme

def train_loop(cfgs, current_epoch, dataloader, model, loss_fn, optimizer, name="Train"):
    loss_dict = defaultdict(lambda: 0)

    if cfgs.lossw_enabled:
        loss_weight = torch.zeros((cfgs.batch_size, 70))
        all_weights = [
                cfgs.w_jaw, 
                cfgs.w_leyeb, cfgs.w_reyeb, 
                cfgs.w_nose, cfgs.w_nosetip, 
                cfgs.w_leye, cfgs.w_reye, 
                cfgs.w_mount, cfgs.w_purpil
            ]
        for idx, (b, e) in enumerate(LMK_PARTS):
            loss_weight[:, b:e] = all_weights[idx]
        loss_weight = loss_weight.to(cfgs.device)

    num_batches = len(dataloader)
    pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, bar_format=TQDM_BAR_FORMAT)

    model.train()
    for batch, (x, y) in pbar:
        x_device = x.to(cfgs.device, non_blocking=True)
        y_device = y.to(cfgs.device, non_blocking=True)

        pred = model(x_device)

        if cfgs.aux_pose:
            loss = loss_fn(pred, y_device[:,:-1])

            if cfgs.lossw_enabled:
                loss[:, :70] *= loss_weight[:loss.shape[0], :]

            pose_weight = y_device[:,-1:]
            loss[:,-3:] *= cfgs.aux_pose_weight * pose_weight
        else:
            loss = loss_fn(pred, y_device)

        total_loss = loss.mean()

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_dict["total"] += total_loss.item()
        loss_dict["nme"] += nme(pred[:,:68*2], y_device[:,:68*2]).mean()
        for (b, e), pname in zip(LMK_PARTS, LMK_PART_NAMES):
            loss_dict[pname] += loss[:, b*2:e*2].mean().item()

        losses_str = f"Total: {loss_dict['total']/(batch+1):>7f}, nmei: {loss_dict['nme']/(batch+1):>7f}"
        for n in LMK_PART_NAMES:
            losses_str = f"{losses_str}, {n}: {loss_dict[n]/(batch+1):>7f}"

        if cfgs.aux_pose:
            loss_dict["pose"] += (loss[:,-3:] / cfgs.aux_pose_weight).mean().item()
            losses_str = f"{losses_str}, pose: {loss_dict['pose']/(batch+1):>7f}"

        pbar.set_description(f"{name} epoch [{current_epoch+1:3d}/{cfgs.epoch:3d}] {losses_str}")
        
        if current_epoch == 0 and batch < cfgs.dump_batch:
            save_batch_png = cfgs.save_dir / f'{name}_batch{batch}.png'
            render_batch(x.cpu().detach().numpy(), y[:,:70*2].cpu().detach().numpy(), save_batch_png)

    for k in loss_dict.keys():
        loss_dict[k] /= num_batches

    return loss_dict