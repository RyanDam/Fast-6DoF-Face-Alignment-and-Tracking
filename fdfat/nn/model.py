import torch
from torch import nn

from .conv import *
from . import module
from . import module2
from . import module3
class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def is_fused(self):
        return False

class LightWeightModel(BaseModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False):
        super().__init__()
        self.pose_rotation = pose_rotation

        # backbone, output size = size/4
        self.backbone = module.LightWeightBackbone(muliplier=muliplier)

        # mainstream, output size:
        # x1 = size/8
        # x2 = size/16
        # x3 = size/16
        self.mainstream = module.MainStreamModule(muliplier=muliplier)

        if pose_rotation:
            self.aux = module.AuxiliaryBackbone(int(16*muliplier), 3, muliplier=muliplier)
            
        output_ch = int((64 + 64 + 64)*muliplier)
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
    
class LightWeightModelR2(BaseModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False):
        super().__init__()
        self.pose_rotation = pose_rotation

        # backbone, output size = size/4
        self.backbone = module2.LightWeightBackbone(muliplier=muliplier)

        # mainstream, output size:
        # x1 = size/8
        # x2 = size/16
        # x3 = size/16
        self.mainstream = module2.MainStreamModule(muliplier=muliplier)

        if pose_rotation:
            self.aux = module2.AuxiliaryBackbone(int(16*muliplier), 3, muliplier=muliplier)
            
        output_ch = int((128 + 128 + 128)*muliplier)
        self.logit = module2.FERegress(output_ch, 70*2)

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

class LightWeightModelR3(BaseModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False):
        super().__init__()
        self.pose_rotation = pose_rotation

        # backbone, output size = size/4
        self.backbone = module3.LightWeightBackbone(muliplier=muliplier)

        # mainstream, output size:
        # x1 = size/8
        # x2 = size/16
        # x3 = size/16
        self.mainstream = module3.MainStreamModule(muliplier=muliplier)

        if pose_rotation:
            self.aux = module3.AuxiliaryBackbone(int(16*muliplier), 3, muliplier=muliplier)
            
        output_ch = int((128 + 128 + 128)*muliplier)
        self.logit = module3.FERegress(output_ch, 70*2)

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

class LandmarkModel(BaseModel):

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False):
        super().__init__()
        
        self.stack1 = nn.Sequential(
            nn.Conv2d( 3, int(32*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(32*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(32*muliplier), int(32*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(32*muliplier), track_running_stats=False),
            nn.ReLU()
        )

        self.stack2 = nn.Sequential(
            nn.Conv2d(int(32*muliplier), int(64*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(64*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(64*muliplier), int(64*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(64*muliplier), track_running_stats=False),
            nn.ReLU()
        )

        self.stack3 = nn.Sequential(
            nn.Conv2d(int(64*muliplier), int(128*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(128*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(128*muliplier), int(128*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(128*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(128*muliplier), int(128*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(128*muliplier), track_running_stats=False),
            nn.ReLU()
        )

        self.stack4 = nn.Sequential(
            nn.Conv2d(int(128*muliplier), int(256*muliplier), 3, stride=2, padding=2),
            nn.BatchNorm2d(int(256*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(256*muliplier), int(256*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(256*muliplier), track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(int(256*muliplier), int(256*muliplier), 3, padding=1),
            nn.BatchNorm2d(int(256*muliplier), track_running_stats=False),
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
    