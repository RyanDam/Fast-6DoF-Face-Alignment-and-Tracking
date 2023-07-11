import tqdm
import torch
from collections import defaultdict

from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES, render_batch
from landmark import TQDM_BAR_FORMAT
from fdfat.metric.metric import nme

def val_loop(cfgs, current_epoch, dataloader, model, loss_fn, name="Valid"):
    loss_dict = defaultdict(lambda: 0)

    num_batches = len(dataloader)
    pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, bar_format=TQDM_BAR_FORMAT)

    model.eval()
    with torch.no_grad():
        for batch, (x, y) in pbar:
            x_device = x.to(cfgs.device, non_blocking=True)
            y_device = y.to(cfgs.device, non_blocking=True)

            pred = model(x_device)

            if cfgs.aux_pose:
                loss = loss_fn(pred, y_device[:,:-1])
                pose_weight = y_device[:,-1:]
                loss[:,-3:] *= cfgs.aux_pose_weight * pose_weight
            else:
                loss = loss_fn(pred, y_device)

            total_loss = loss.mean()

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