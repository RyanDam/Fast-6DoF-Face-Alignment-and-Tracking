import tqdm
import torch
import numpy as np
from collections import defaultdict

from fdfat.utils.utils import render_batch
from fdfat import TQDM_BAR_FORMAT
from fdfat.metric.metric import nme
from fdfat.utils.model_utils import normalize_tensor

def val_loop(cfgs, current_epoch, dataloader, model, loss_fn, face_loss_fn=None, name="Valid"):

    with torch.no_grad():
        loss_dict = defaultdict(lambda: 0)
        nme_list = []

        num_batches = len(dataloader)
        pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, bar_format=TQDM_BAR_FORMAT)

        model.eval()
        for batch, data in pbar:
            x_device = data["img"].to(cfgs.device, non_blocking=True)
            y_device = data["landmark"].to(cfgs.device, non_blocking=True)
            face_clss = data["is_face"].to(cfgs.device, non_blocking=True)

            if not cfgs.pre_norm:
                x_device = normalize_tensor(x_device).type(torch.float32)

            pred = model(x_device)

            # calculate loss
            main_loss = loss_fn(pred[:,:cfgs.lmk_num*2], y_device[:,:cfgs.lmk_num*2])*face_clss
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

            loss_dict["total"] += total_loss.item()
            nme_val = nme(pred[:,:68*2], y_device[:,:68*2], reduced=False)[face_clss[:,0]>0.5,...].cpu().detach().numpy()
            loss_dict["nme"] += nme_val.mean()
            nme_list.append(nme_val)

            for (b, e), pname in zip(cfgs.lmk_parts, cfgs.lmk_part_names):
                loss_dict[pname] += main_loss[:, b*2:e*2].mean().item()

            losses_str = f"Total: {loss_dict['total']/(batch+1):>7f}, nmei: {loss_dict['nme']/(batch+1):>7f}"
            for n in cfgs.lmk_part_names:
                losses_str = f"{losses_str}, {n}: {loss_dict[n]/(batch+1):>7f}"

            if cfgs.aux_pose:
                loss_dict["pose"] += (pose_loss / cfgs.aux_pose_weight).mean().item()
                losses_str = f"{losses_str}, pose: {loss_dict['pose']/(batch+1):>7f}"

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

        for k, v in loss_dict.items():
            loss_dict[k] = float(v)

        nme_list = np.concatenate(nme_list)

        loss_dict["nme_stat"] ={
            "nme_min": float(np.min(nme_list)),
            "nme_max": float(np.max(nme_list)),
            "nme_mean": float(np.mean(nme_list)),
            "nme_median": float(np.median(nme_list)),
            "nme_std": float(np.std(nme_list)),
            "nme_part": {
                "nme_min": np.min(nme_list, axis=0).tolist(),
                "nme_max": np.max(nme_list, axis=0).tolist(),
                "nme_mean": np.mean(nme_list, axis=0).tolist(),
                "nme_median": np.median(nme_list, axis=0).tolist(),
                "nme_std": np.std(nme_list, axis=0).tolist(),
            }
        }
        
        return loss_dict