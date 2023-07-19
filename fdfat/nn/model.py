import torch
from torch import nn

from fdfat.utils.model_utils import intersect_dicts
from fdfat.utils.logger import LOGGER

from .conv import *
from . import module

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def is_fused(self):
        return False
    
    def load(self, weights, verbose=True):
        """Load the weights into the model.

        Args:
            weights (dict) or (torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.state_dict())} items from pretrained weights')

class LightWeightModel(BaseModel):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False, act=True):
        super().__init__()
        self.pose_rotation = pose_rotation

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # backbone, output size = size/4
        self.backbone = module.LightWeightBackbone(muliplier=muliplier, act=self.act)

        # mainstream, output size:
        # x1 = size/8
        # x2 = size/16
        # x3 = size/16
        self.mainstream = module.MainStreamModule(muliplier=muliplier, act=self.act)

        if pose_rotation:
            self.aux = module.AuxiliaryBackbone(int(16*muliplier), 3, muliplier=muliplier, act=self.act)
            
        output_ch = int((128 + 128 + 128)*muliplier)
        self.logit = module.FERegress(output_ch, 70*2)

        self.concat = Concat()

        initialize_weights(self)

    def forward(self, x):

        x = self.backbone(x)
        
        if self.pose_rotation:
            aux = self.aux(x)

        x = self.mainstream(x)
        
        x = self.logit(x)

        if self.pose_rotation:
            x = self.concat([x, aux])

        return x

class LightWeightModelSiLU(LightWeightModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False, act=True):
        super().__init__(imgz=imgz, muliplier=muliplier, pose_rotation=pose_rotation, act=nn.SiLU())

class LandmarkModel(BaseModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False):
        super().__init__()
        
        self.stack1 = nn.Sequential(
            nn.Conv2d( 3, int(32*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(32*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(32*muliplier), int(32*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(32*muliplier)),
            nn.ReLU()
        )

        self.stack2 = nn.Sequential(
            nn.Conv2d(int(32*muliplier), int(64*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(64*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(64*muliplier), int(64*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(64*muliplier)),
            nn.ReLU()
        )

        self.stack3 = nn.Sequential(
            nn.Conv2d(int(64*muliplier), int(128*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(128*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(128*muliplier), int(128*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(128*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(128*muliplier), int(128*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(128*muliplier)),
            nn.ReLU()
        )

        self.stack4 = nn.Sequential(
            nn.Conv2d(int(128*muliplier), int(256*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(256*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(256*muliplier), int(256*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(256*muliplier)),
            nn.ReLU(),
            nn.Conv2d(int(256*muliplier), int(256*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(256*muliplier)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.stack5 = nn.Sequential(
            nn.Linear(int(256*muliplier), int(192*muliplier)),
            nn.Tanh(),
            nn.Linear(int(192*muliplier), 70*2),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = x.flatten(start_dim=1)
        x = self.stack5(x)
        return x
    