import os
import random
import platform
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision

from fvcore.nn import FlopCountAnalysis

from fdfat.utils.logger import LOGGER, check_version

TORCHVISION_0_10 = check_version(torchvision.__version__, '0.10.0')
TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_1_11 = check_version(torch.__version__, '1.11.0')
TORCH_1_12 = check_version(torch.__version__, '1.12.0')
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def model_info(model, detailed=False, verbose=True, imgsz=640, device="cpu"):
    """Model information. imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]."""
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)  # number gradients
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s' %
                        (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))

    flops = get_flops(model, imgsz, device=device)
    fused = ' (fused)' if model.is_fused() else ''
    fs = f', {flops:.1f} GFLOPs' if flops else ''
    LOGGER.info(f'Summary{fused}: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')
    return n_p, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640, device="cpu"):
    im = torch.empty((1, 3, imgsz, imgsz)).to(device)  # input image in BCHW format
    flops = FlopCountAnalysis(model, im)
    return flops.total()*10e-9
    # """Return a YOLO model's FLOPs."""
    # try:
    #     model = de_parallel(model)
    #     p = next(model.parameters())
    #     stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
    #     im = torch.empty((1, p.shape[1], imgsz, imgsz), device=p.device)  # input image in BCHW format
    #     print(im.shape)
    #     flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1E9 * 2 if thop else 0  # stride GFLOPs
    #     imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    #     flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    #     return flops
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     print(e)
    #     return 0
    
def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)  # for Multi-GPU, exception safe
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            LOGGER.warning('WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.')

def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = f'Realtime landmark 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() and TORCH_2_0:
        # Prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    return torch.device(arg)

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def preprocess(img_pil, imgsz):
    img_pil = img_pil.resize((imgsz, imgsz))
    img_np = normalize_tensor(np.array(img_pil))
    img_np = np.transpose(img_np, [2, 0, 1])[np.newaxis, ...]
    return img_np

def normalize_tensor(tensor):
    return tensor/127.5 - 1
    # return tensor/255
    # return (tensor/255 - 0.44531356896770125 ) / 0.2692461874154524

def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1  # opset