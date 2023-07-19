import os
from pathlib import Path
import shutil
import pandas as pd
import math

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from fdfat.utils.pose_estimation import MODEL_3D_POINTS

LMK_PARTS = [
    [0, 17], # jaw
    [17, 22], # left eye brown
    [22, 27], # right eye brown
    [27, 31], # nose
    [31, 36], # nose tip
    [36, 42], # left eye
    [42, 48], # right eye
    [48, 68], # mount
    [68, 70], # purpils
]

LMK_PART_NAMES = [
    "jaw", "leyeb", "reyeb", "nose", "nosetip", "leye", "reye", "mount", "purpils"
]

def circle(cx, cy, R=2):
    return cx - R//2, cy - R//2, cx + R//2, cy + R//2

def render_lmk_nme(nme, imgsz=1024, base_radius=200, title=None):

    # load model
    model_points = np.array(MODEL_3D_POINTS, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    model_points[:, 2] *= -1

    # normalize
    minx, miny, maxx, maxy = model_points[:,0].min(), model_points[:,1].min(), model_points[:,0].max(), model_points[:,1].max()
    lmk_model = model_points[:,:2].copy()
    lmk_model[:,0] = (lmk_model[:,0] - minx)/(maxx-minx)
    lmk_model[:,1] = (lmk_model[:,1] - miny)/(maxy-miny)

    target = Image.new('RGB', (imgsz, imgsz), color=(255, 255, 255))
    draw = ImageDraw.Draw(target)

    lmk_image = lmk_model * (imgsz - 144) + 72
    for (x, y), v in zip(lmk_image, nme):
        target_radius = 200*v
        draw.ellipse(circle(x, y, R=target_radius), fill=(201, 168, 75))

        textbox = draw.textbbox((x, y+target_radius//2 + 2), f"{v*100:3.2f}")
        text_width = textbox[2]-textbox[0]
        draw.text((x-text_width//2, y+target_radius//2 + 2), f"{v*100:3.2f}", fill=(0, 0, 0))

    if title is not None:
        draw.text((10, 10), title, fill=(0, 0, 0))

    return target

def render_lmk(img, lmk, point_size=2, render=False):

    draw = ImageDraw.Draw(img)

    for begin, end in LMK_PARTS[:-1]:
        lx, ly = lmk[begin]
        for idx in range(begin+1, end):
            x, y = lmk[idx]
            draw.line([lx, ly, x, y], width=2, fill=(255, 0, 0))
            lx, ly = x, y

    for x, y in lmk:
        draw.rectangle([x-point_size/2, y-point_size/2, x+point_size/2, y+point_size/2], fill=(255, 255, 0))

    # bbox = gen_bbox(lmk)

    # print(bbox.flatten().astype(np.int32).tolist())
    # draw.rectangle(bbox.flatten().astype(np.int32).tolist(), width=2, outline=(0, 255, 255))

    if render:
        plt.imshow(img)
        plt.show()

    return img


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def generate_graph(csv_path, save_path, highlight_total=True):
    df = pd.read_csv(csv_path, sep="\t")
    all_fields = ["total", *LMK_PART_NAMES]
    epoch_idxes = df["epoch"].to_numpy()

    fig = plt.figure(figsize=(10,20))
    (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=True, sharey=True)

    for f in all_fields:
        lw = 3 if highlight_total and f == "total" else 1
        ax1.plot(epoch_idxes, df[f].to_numpy(), label=f, linewidth=lw)
    ax1.legend()
    ax1.set_title("train")

    for f in all_fields:
        lw = 3 if highlight_total and f == "total" else 1
        ax2.plot(epoch_idxes, df[f"test_{f}"].to_numpy(), label=f, linewidth=lw)
    ax2.legend()
    ax2.set_title("val")
    
    ax3.plot(epoch_idxes, df[f"total"].to_numpy(), label="train", linewidth=3)
    ax3.plot(epoch_idxes, df[f"test_total"].to_numpy(), label="val", linewidth=3)
    ax3.legend()
    ax3.set_title("train + val")

    fig.savefig(save_path, bbox_inches='tight')
    plt.close()

def render_batch(batchx, batchy, save_path):
    batch_size = batchx.shape[0]

    grid_w = int(math.sqrt(batch_size)) + 1

    fig = plt.figure(figsize=(20,20))
    axes = fig.subplots(grid_w, grid_w, sharex=True, sharey=True, squeeze=False)
    for idx in range(batch_size):
        img_np = (batchx[idx,...].transpose([1, 2, 0]) * 127.5) + 127.5
        img = Image.fromarray(img_np.astype(np.uint8))
        lmk = (batchy[idx,:].reshape(70,2) + 0.5) * img.size[0]
        rendered = render_lmk(img, lmk)

        x = int(idx/grid_w)
        y = idx % grid_w
        axes[x, y].imshow(rendered)
        axes[x, y].axis('off')

    fig.savefig(save_path, bbox_inches='tight')
    plt.close()

def read_file_list(p, base_path=None):
    base_path = base_path if base_path is not None else ""
    with open(p, "r") as f:
        d = f.readlines()
        d = [os.path.join(base_path, a.strip("\n")) for a in d]
    return d
