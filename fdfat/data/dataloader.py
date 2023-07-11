
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import cv2

from fdfat.utils.pose_estimation import PoseEstimator

POSE_ROTAION_MED = np.array([-0.02234655,  0.28259986, -2.98499613])
POSE_ROTATION_STD = np.array([0.87680358, 0.53386852, 0.25789746])

def gen_bbox(lmk, scale=[1.4, 1.6], offset=0.2, square=True):
    bbox = np.array([np.min(lmk, 0), np.max(lmk, 0)])
    size = bbox[1, :] - bbox[0, :]
    if square:
        m = np.max(size)
        size[0] = m
        size[1] = m
    center = (bbox[1, :] + bbox[0, :])/2

    offset_ab = (2*np.random.random(2)-1)*offset*size

    size = size*np.random.uniform(low=scale[0], high=scale[1], size=1)
    center = center + offset_ab

    return np.vstack([center-size/2, center+size/2])

def read_data(img_path, lmk_scale=1.0, aug=None, imgsz=128):
    img = Image.open(img_path)
    lmk_path = img_path.replace(".png", "_ldmks.txt")
    with open(lmk_path, 'r') as f:
        lmk = f.readlines()
        lmk = np.array([[float(n) for n in l.strip("\n").split(" ")] for l in lmk])

    bbox = gen_bbox(lmk)

    croped = np.array(img.crop(bbox.astype(np.int32).flatten().tolist()))
    lmk[:, 0] -= bbox[0, 0]
    lmk[:, 1] -= bbox[0, 1]

    if aug is not None:
        transformed = aug(image=croped, keypoints=lmk)
        croped = transformed['image']
        lmk = transformed['keypoints']
        lmk = np.array(lmk)

    croped = (croped / 127.5) - 1

    # normalize
    lmk /= imgsz
    lmk -= 0.5
    lmk *= lmk_scale

    # bbox_center = (bbox[0, :] + bbox[1, :]) / 2
    # lmk[:,0] -= bbox_center[0]
    # lmk[:,1] -= bbox_center[1]
    # lmk[:,0] /= croped.size[0]
    # lmk[:,1] /= croped.size[1]
    # lmk[:,0] *= lmk_scale
    # lmk[:,1] *= lmk_scale
    # lmk = lmk.astype(np.float32)

    return croped.astype(np.float32), lmk.astype(np.float32)

class LandmarkDataset(Dataset):
    def __init__(self, cfgs, annotations_files, imgsz=128, aug=True, pose_rotation=False):
        self.img_paths = annotations_files
        self.imgsz = imgsz
        self.pose_rotation = pose_rotation

        if aug:
            self.aug = A.Compose([
                # A.HorizontalFlip(p=0.5), # wrong lmk idx
                A.ToGray(p=0.2),
                A.Rotate (limit=10, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.HueSaturationValue(p=0.5), 
                    A.RGBShift(p=0.7)
                ], p=0.1),
                A.OneOf([
                    A.Defocus(p=0.1),
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.1),
                A.ISONoise(p=0.2),
                A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
                A.Resize(imgsz, imgsz)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.aug = A.Compose([
                A.Resize(imgsz, imgsz)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        self.trans = transforms.ToTensor()

        print(f"Loaded {len(annotations_files)} samples")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img, lmk = read_data(img_path, aug=self.aug, imgsz=self.imgsz)
        
        if self.pose_rotation:
            estimator = PoseEstimator(self.imgsz, self.imgsz)
            lmk_denorm = (lmk + 0.5)*self.imgsz
            rot_vec, _ = estimator.solve(lmk_denorm[:68,:])
            rot_vec_norm = np.divide(rot_vec[:,0] - POSE_ROTAION_MED, POSE_ROTATION_STD*2)

            rot_weight = 1
            if np.abs(rot_vec_norm).max() > 1.5:
                rot_weight = 0
            lmk_rot_norm = np.concatenate([lmk.flatten().astype(np.float32), rot_vec_norm.astype(np.float32), np.array([rot_weight], dtype=np.float32)])
            return img.transpose((2, 0, 1)), lmk_rot_norm.flatten()
        else:
            return img.transpose((2, 0, 1)), lmk.flatten()