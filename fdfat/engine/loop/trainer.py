import tqdm
import torch
import numpy as np
from collections import defaultdict

from fdfat.utils.utils import render_batch
from fdfat.utils.profiler import Profile
from fdfat import TQDM_BAR_FORMAT
from fdfat.metric.metric import nme
from fdfat.data.dataloader import LMK_POINT_MEANS
from fdfat.utils.model_utils import normalize_tensor

def train_loop(cfgs, current_epoch, dataloader, model, loss_fn, optimizer, face_loss_fn=None, name="Train"):
    loss_dict = defaultdict(lambda: 0)
    nme_list = []

    if cfgs.lossw_enabled:
        loss_weight = torch.zeros((cfgs.batch_size, cfgs.lmk_num*2))
        all_weights = [
                cfgs.w_jaw, 
                cfgs.w_leyeb, cfgs.w_reyeb, 
                cfgs.w_nose, cfgs.w_nosetip, 
                cfgs.w_leye, cfgs.w_reye, 
                cfgs.w_mount, cfgs.w_purpil
            ]
        for idx, (b, e) in enumerate(cfgs.lmk_parts):
            loss_weight[:, b*2:e*2] = all_weights[idx]
        loss_weight = loss_weight.to(cfgs.device)

    # if cfgs.lmk_mean:
    #     lmk_means = torch.from_numpy(LMK_POINT_MEANS)

    num_batches = len(dataloader)
    pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, bar_format=TQDM_BAR_FORMAT)

    model.train()
    if cfgs.freeze_landmark: model.freeze()
    for batch, data in pbar:
        x_device = data["img"].to(cfgs.device, non_blocking=True)
        y_device = data["landmark"].to(cfgs.device, non_blocking=True)
        face_clss = data["is_face"].to(cfgs.device, non_blocking=True)

        if not cfgs.pre_norm:
            x_device = normalize_tensor(x_device).type(torch.float32)

        pred = model(x_device)

        # calculate loss
        main_loss = loss_fn(pred[:,:cfgs.lmk_num*2], y_device[:,:cfgs.lmk_num*2])*face_clss
        if cfgs.lossw_enabled:
            main_loss[:, :cfgs.lmk_num*2] *= loss_weight[:main_loss.shape[0], :cfgs.lmk_num*2]
        total_loss = main_loss.mean()
        loss_devide = 1

        if cfgs.aux_pose:
            pose_rotation = data["pose_rotation"].to(cfgs.device, non_blocking=True)
            pose_weight = data["pose_weight"].to(cfgs.device, non_blocking=True)

            pose_loss = loss_fn(pred[:,-3:], pose_rotation)*face_clss
            pose_loss *= cfgs.aux_pose_weight * pose_weight

            total_loss = total_loss + pose_loss.mean()
            loss_devide += 1

        if cfgs.face_cls:
            face_loss = face_loss_fn(pred[:,cfgs.lmk_num*2:cfgs.lmk_num*2+1], face_clss)
            face_loss *= cfgs.face_cls_weight

            total_loss = total_loss + face_loss.mean()
            loss_devide += 1

            face_cls_correct = ((pred[:,cfgs.lmk_num*2:cfgs.lmk_num*2+1] >= 0.5) == face_clss).type(torch.float).mean().item()

        total_loss /= loss_devide

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_dict["total"] += total_loss.item()
        # if cfgs.lmk_mean:
        #     nme_val = nme(pred[:,:68*2], y_device[:,:68*2]).mean().cpu().detach().numpy()
        # else:
        nme_val = nme(pred[:,:68*2], y_device[:,:68*2])[face_clss[:,0]>0.5].mean().cpu().detach().numpy()
        loss_dict["nme"] += nme_val
        nme_list.append(nme_val)

        for (b, e), pname in zip(cfgs.lmk_parts, cfgs.lmk_part_names):
            loss_dict[pname] += main_loss[:, b*2:e*2].mean().item()

        losses_str = f"Total: {loss_dict['total']/(batch+1):>7f}, nmei: {loss_dict['nme']/(batch+1):>7f}"
        for n in cfgs.lmk_part_names:
            losses_str = f"{losses_str}, {n}: {loss_dict[n]/(batch+1):>7f}"

        if cfgs.aux_pose:
            loss_dict["pose"] += (pose_loss / cfgs.aux_pose_weight).mean().item()
            losses_str = f"{losses_str} pose: {loss_dict['pose']/(batch+1):>7f}"

        if cfgs.face_cls:
            loss_dict["face"] += (face_loss / cfgs.face_cls_weight).mean().item()
            losses_str = f"{losses_str} face: {loss_dict['face']/(batch+1):>7f}"

            loss_dict["face_acc"] += face_cls_correct
            losses_str = f"{losses_str} face_acc: {loss_dict['face_acc']/(batch+1):>7f}"

        pbar.set_description(f"{name} epoch [{current_epoch+1:3d}/{cfgs.epoch:3d}] {losses_str}")

        if current_epoch == 0 and batch < cfgs.dump_batch:
            save_batch_png = cfgs.save_dir / f'{name}_batch{batch}.png'
            render_batch(x_device.cpu().detach().numpy(), y_device[:,:70*2].cpu().detach().numpy(), save_batch_png)

    for k in loss_dict.keys():
        loss_dict[k] /= num_batches

    nme_list = np.array(nme_list)
    loss_dict["nme_min"] = np.min(nme_list)
    loss_dict["nme_max"] = np.max(nme_list)
    loss_dict["nme_mean"] = np.mean(nme_list)
    loss_dict["nme_median"] = np.median(nme_list)
    loss_dict["nme_std"] = np.std(nme_list)

    return loss_dict