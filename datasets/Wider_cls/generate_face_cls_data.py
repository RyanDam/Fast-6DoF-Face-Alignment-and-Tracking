import os
import tqdm
import random
import numpy as np
from PIL import Image

from fdfat.utils.box_utils import to_landmark_box, iou_of, iou_batch, guard_bbox_inside

data_path = "<path_to_data.txt>"

with open(data_path, 'r') as f:
    all_image_path = f.readlines()
    all_image_path = [a.strip("\n") for a in all_image_path]

def random_box(imw, imh, scale=0.3, ratio=1, mins=64):

    w = random.randint(mins, int(imw*scale))
    # h = random.randint(mins, int(imh*scale))
    h = min(w*ratio, imh)

    x = random.randint(0, imw-w)
    y = random.randint(0, imh-h)

    return x, y, x+w, y+h

def random_transform_box(box, translate=(-0.2, 0.2), scale=(0.8, 1.1)):

    x, y, xx, yy = box
    cx, cy = (xx+x)/2, (yy+y)/2
    w, h = xx-x, yy-y

    tx = w*(random.random()*(translate[1]-translate[0]) + translate[0])
    ty = h*(random.random()*(translate[1]-translate[0]) + translate[0])

    nw = w*(random.random()*(scale[1]-scale[0]) + scale[0])
    nh = h*(random.random()*(scale[1]-scale[0]) + scale[0])

    ncx, ncy = cx+tx, cy+ty

    return np.array([ncx - nw/2, ncy - nh/2, ncx + nw/2, ncy + nh/2]).astype(np.int32)

def read_data(imgp, start_num_0, start_num_1):
    
    target_save = "<save_folder>"

    anno_path = imgp.replace("/images/", "/labels/").replace(".jpg", ".txt")

    img = Image.open(imgp)

    with open(anno_path, 'r') as f:
        annos = f.readlines()
        annos = [a.strip("\n").split(" ") for a in annos]
        annos = np.array([[float(a) for a in b if a != ""] for b in annos])
        annos = annos[:,1:]
        
    imw, imh = img.size
    annos[:,0] *= imw
    annos[:,1] *= imh
    annos[:,2] *= imw
    annos[:,3] *= imh

    annos[:,0] -= annos[:,2]/2
    annos[:,1] -= annos[:,3]/2
    annos[:,2] = annos[:,0] + annos[:,2]
    annos[:,3] = annos[:,1] + annos[:,3]

    num_1 = start_num_1 + 1

    annos_landmark = [to_landmark_box(b) for b in annos]
    for b in annos_landmark:
        try:
            for i in range(3):
                b = guard_bbox_inside(random_transform_box(b) if i > 0 else b, imw, imh)
                s = max(b[2]-b[0], b[3]-b[1])
                if s < 72: continue
                img.crop(b).save(os.path.join(target_save, "1", f"{num_1:08d}.jpg"))
                num_1 += 1
        except:
            continue

    num_0 = start_num_0 + 1
    for _ in range(5):
        try:
            b = guard_bbox_inside(random_box(imw, imh), imw, imh)
            iou = iou_batch([b], annos_landmark)
            max_iou = iou.max()
            if max_iou < 0.3:
                img.crop(b).save(os.path.join(target_save, "0", f"{num_0:08d}.jpg"))
                num_0 += 1
        except:
            continue
        
    return num_0, num_1

num_0, num_1 = 0, 0
for p in tqdm.tqdm(all_image_path):
    num_0, num_1 = read_data(p, num_0, num_1)
