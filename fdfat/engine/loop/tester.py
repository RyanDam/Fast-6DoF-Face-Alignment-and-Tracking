import tqdm
import torch
from collections import defaultdict

from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES, render_batch
from fdfat import TQDM_BAR_FORMAT
from fdfat.metric.metric import nme

def test_loop(cfgs, dataloader, model, loss_fn, name="Test"):

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

            nme_loss = nme(pred[:,:68*2], y_device[:,:68*2], reduced=False)

            loss_dict["nme_mean_parts"] += nme_loss.mean(dim=0)

            loss_dict["nme"] += nme_loss.mean()
            for (b, e), pname in zip(LMK_PARTS, LMK_PART_NAMES):
                loss_dict[pname] += loss[:, b*2:e*2].mean().item()

            losses_str = f"Total: {loss_dict['total']/(batch+1):>7f}, nmei: {loss_dict['nme']/(batch+1):>7f}"
            for n in LMK_PART_NAMES:
                losses_str = f"{losses_str}, {n}: {loss_dict[n]/(batch+1):>7f}"

            if cfgs.aux_pose:
                loss_dict["pose"] += (loss[:,-3:] / cfgs.aux_pose_weight).mean().item()
                losses_str = f"{losses_str}, pose: {loss_dict['pose']/(batch+1):>7f}"

            pbar.set_description(f"{name} {losses_str}")

    for k in loss_dict.keys():
        loss_dict[k] /= num_batches

    return loss_dict