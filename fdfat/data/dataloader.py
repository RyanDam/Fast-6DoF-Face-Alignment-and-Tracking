

import tqdm
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from copy import deepcopy

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

from fdfat.utils.pose_estimation import PoseEstimator
from fdfat.utils.model_utils import normalize_tensor
from fdfat.utils.logger import LOGGER

POSE_ROTAION_MED = np.array([-0.02234655,  0.28259986, -2.98499613])
POSE_ROTATION_STD = np.array([0.87680358, 0.53386852, 0.25789746])
LMK_POINT_MEANS = np.array([-0.27238744497299194,
 -0.18219642341136932,
 -0.25812533497810364,
 -0.08848361670970917,
 -0.24631524085998535,
 -0.013254939578473568,
 -0.2338235080242157,
 0.06554146856069565,
 -0.2148175835609436,
 0.13076455891132355,
 -0.17676620185375214,
 0.19475635886192322,
 -0.12203802913427353,
 0.24699880182743073,
 -0.061511386185884476,
 0.2886599600315094,
 0.007390471175312996,
 0.3102368116378784,
 0.07749525457620621,
 0.288831889629364,
 0.13470110297203064,
 0.25185081362724304,
 0.1884017288684845,
 0.1998489946126938,
 0.22290366888046265,
 0.13987477123737335,
 0.2392514944076538,
 0.07432771474123001,
 0.2506774067878723,
 -0.006450130138546228,
 0.25963759422302246,
 -0.07902003824710846,
 0.270471453666687,
 -0.17388933897018433,
 -0.20455655455589294,
 -0.2547716796398163,
 -0.17472998797893524,
 -0.2664816081523895,
 -0.1358528882265091,
 -0.2704051733016968,
 -0.09964480251073837,
 -0.2644731104373932,
 -0.06238285079598427,
 -0.2496853470802307,
 0.06375256180763245,
 -0.2492913156747818,
 0.10013817250728607,
 -0.2625551223754883,
 0.13949207961559296,
 -0.2664727568626404,
 0.17374610900878906,
 -0.26295679807662964,
 0.20365028083324432,
 -0.2488793581724167,
 -0.0005358753842301667,
 -0.19838227331638336,
 -0.00020459208462852985,
 -0.13754519820213318,
 0.00037407688796520233,
 -0.08316519856452942,
 0.0017506529111415148,
 -0.021848831325769424,
 -0.06879628449678421,
 0.0066255987621843815,
 -0.03807201236486435,
 0.03021293692290783,
 0.002773846033960581,
 0.039460308849811554,
 0.04299839958548546,
 0.033025112003088,
 0.07343906164169312,
 0.0057152207009494305,
 -0.16360820829868317,
 -0.18467222154140472,
 -0.14117562770843506,
 -0.19985820353031158,
 -0.10185961425304413,
 -0.1969078779220581,
 -0.07602246850728989,
 -0.1753842830657959,
 -0.10300877690315247,
 -0.165273517370224,
 -0.14242322742938995,
 -0.1653825342655182,
 0.07576962560415268,
 -0.17290200293064117,
 0.10372519493103027,
 -0.1935623437166214,
 0.14142119884490967,
 -0.193919837474823,
 0.16329725086688995,
 -0.1780003309249878,
 0.1437627524137497,
 -0.15984562039375305,
 0.10441101342439651,
 -0.16048423945903778,
 -0.09351976215839386,
 0.12908431887626648,
 -0.07556243240833282,
 0.10674221068620682,
 -0.04644044488668442,
 0.09543296694755554,
 0.003641054965555668,
 0.09507127851247787,
 0.05327253043651581,
 0.09620969742536545,
 0.08406565338373184,
 0.1084865853190422,
 0.10406434535980225,
 0.13132770359516144,
 0.08168935030698776,
 0.16024908423423767,
 0.04768916592001915,
 0.18506872653961182,
 0.006121656857430935,
 0.19179826974868774,
 -0.03787531331181526,
 0.18208207190036774,
 -0.07063888013362885,
 0.15789297223091125,
 -0.08589783310890198,
 0.1289578080177307,
 -0.04514375701546669,
 0.11801735311746597,
 0.004851692356169224,
 0.12016500532627106,
 0.05463751032948494,
 0.11946801096200943,
 0.09600751101970673,
 0.13059113919734955,
 0.046982888132333755,
 0.1538499891757965,
 0.00576262129470706,
 0.1602175533771515,
 -0.03772566467523575,
 0.15227624773979187,
 -0.12288659065961838,
 -0.19115349650382996,
 0.1218595802783966,
 -0.1859482377767563])

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

def read_raw_lmk(lmk_path):

    with open(lmk_path, 'r') as f:
        lmk = f.readlines()
        lmk = np.array([[float(n) for n in l.strip("\n").split(" ")] for l in lmk])

    bbox = gen_bbox(lmk)

    return bbox, lmk

class LandmarkDataset(Dataset):

    def __init__(self, cfgs, annotations_files, aug=True, pose_rotation=False, cache_path=None):
        self.lmk_num = cfgs.lmk_num
        self.lmk_mean = cfgs.lmk_mean
        self.norm = cfgs.pre_norm
        self.img_paths = annotations_files
        self.imgsz = cfgs.imgsz
        self.pose_rotation = cfgs.aux_pose
        
        self.cache_path = cache_path
        self.cache = self.read_cache()

        if aug:
            self.aug = A.Compose([
                A.Resize(self.imgsz, self.imgsz),
                # A.HorizontalFlip(p=0.5), # do not use, wrong lmk idx
                # A.Affine(scale=(0.9,1.1), translate_percent=0.1),
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
                A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.aug = A.Compose([
                A.Resize(self.imgsz, self.imgsz)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        self.trans = transforms.ToTensor()

        LOGGER.info(f"Loaded {len(annotations_files)} samples")

    def __len__(self):
        if self.cache is None:
            return len(self.img_paths)
        else:
            return len(self.cache)

    def __getitem__(self, idx):
        if self.cache is None:
            img_path = self.img_paths[idx]
            lmk_path = img_path.replace(".png" if img_path.endswith(".png") else ".jpg", "_ldmks.txt")
            bbox, lmk = read_raw_lmk(lmk_path)
            img = Image.open(img_path).convert("RGB")
        else:
            img_path, _, lmk = deepcopy(self.cache[idx])
            bbox = gen_bbox(lmk) # intergrated translate and scale for better performance
            img = Image.open(img_path).convert("RGB")

        img, lmk = self.preprocess_raw(img, bbox, lmk)
        img = img.transpose((2, 0, 1))

        if self.pose_rotation:
            estimator = PoseEstimator(self.imgsz, self.imgsz)
            lmk_denorm = (lmk + 0.5)*self.imgsz
            rot_vec, _ = estimator.solve(lmk_denorm[:68,:])
            rot_vec_norm = np.divide(rot_vec[:,0] - POSE_ROTAION_MED, POSE_ROTATION_STD*2)

            rot_weight = 1
            if np.abs(rot_vec_norm).max() > 1.5:
                rot_weight = 0
            lmk = np.concatenate([lmk.flatten().astype(np.float32), rot_vec_norm.astype(np.float32), np.array([rot_weight], dtype=np.float32)])

        lmk = lmk.flatten()

        if self.lmk_mean:
            lmk[:self.lmk_num] -= LMK_POINT_MEANS[:self.lmk_num]

        return img, lmk

    def preprocess_raw(self, img, bbox, lmk, lmk_scale=1.0):
        croped = np.array(img.crop(bbox.astype(np.int32).flatten().tolist()))
        lmk[:, 0] -= bbox[0, 0]
        lmk[:, 1] -= bbox[0, 1]

        if self.aug is not None:
            transformed = self.aug(image=croped, keypoints=lmk)
            croped = transformed['image']
            lmk = transformed['keypoints']
            lmk = np.array(lmk)

        # normalize
        lmk /= croped.shape[1]
        lmk -= 0.5
        lmk *= lmk_scale
        lmk = lmk.astype(np.float32)

        if self.norm:
            croped = normalize_tensor(croped)
            croped = croped.astype(np.float32)
        else:
            croped.astype(np.uint8)

        return croped, lmk

    def build_cache(self):
        LOGGER.info(f"Building cache")
        
        cache = []
        for img_path in tqdm.tqdm(self.img_paths):
            lmk_path = img_path.replace(".png" if img_path.endswith(".png") else ".jpg", "_ldmks.txt")
            try:
                bbox, lmk = read_raw_lmk(lmk_path)
                cache.append((img_path, bbox, lmk))
            except Exception as e:
                LOGGER.info(f"Read data error: {lmk_path}:\n{e}")
                
        LOGGER.info(f"Cache loaded: {len(cache)}/{len(self.img_paths)}")
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache, f)
        LOGGER.info(f"Cache saved: {self.cache_path}")

        return cache

    def read_cache(self):
        if self.cache_path is None:
            return None

        self.cache_path = Path(self.cache_path)
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            LOGGER.info(f"Loaded cache ({len(cache)} samples) at {self.cache_path}")
            return cache

        cache = self.build_cache()

        return cache
