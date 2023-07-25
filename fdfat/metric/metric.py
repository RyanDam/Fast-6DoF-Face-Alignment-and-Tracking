import torch
import numpy as np

def nme(pred, gt, left_index=[36], right_index=[45], reduced=True):

    if len(pred.shape) == 1:
        pred_axis = pred.reshape([1, pred.shape[0]//2, 2])
    elif len(pred.shape) == 2:
        pred_axis = pred.reshape([pred.shape[0], pred.shape[1]//2, 2])
    else:
        pred_axis = pred

    if len(gt.shape) == 1:
        gt_axis = gt.reshape([1, gt.shape[0]//2, 2])
    elif len(gt.shape) == 2:
        gt_axis = gt.reshape([gt.shape[0], gt.shape[1]//2, 2])
    else:
        gt_axis = gt

    eye_span = torch.linalg.norm(gt_axis[:, left_index, :] - gt_axis[:, right_index, :], dim=(1,2))
    error = torch.linalg.norm(pred_axis - gt_axis, dim=(2))/eye_span.view(eye_span.shape[0], 1)

    if reduced:
        return error.mean(dim=1)
    else:
        return error