import torch
from torch import nn

from .conv import *

class IdentifyBlock(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, in_ch, out_ch, expand, k=3, act=None, stride=1, dilation_rate=1):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        expand_ch = int(expand*in_ch)

        # expansion
        self.conv1 = Conv(in_ch, expand_ch, k=1, act=self.act)

        # depthwise
        self.conv2 = DWConv(expand_ch, expand_ch, k=k, s=stride, d=dilation_rate, act=self.act)

        # squeeze
        self.conv3 = Conv(expand_ch, out_ch, k=1, act=False)

        self.need_fuse = in_ch == out_ch
        self.need_pool = stride == 2
        if self.need_fuse and self.need_pool:
            self.pool = nn.MaxPool2d(2, stride=2)

        initialize_weights(self)

    def forward(self, x):
        
        ex = self.conv1(x)
        de = self.conv2(ex)
        sq = self.conv3(de)

        if self.need_fuse:
            if self.need_pool:
                x = self.pool(x)
            return sq + x
        else:
            return sq

class LightWeightBackbone(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=None):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # early dropout
        self.conv11 = Conv(3, 16, s=2, act=nn.PReLU())

        self.conv21 = IdentifyBlock(16, int(16*muliplier), expand_rate, stride=2, act=self.act)
        self.conv22 = IdentifyBlock(int(16*muliplier), int(16*muliplier), expand_rate, act=self.act)
        self.conv23 = IdentifyBlock(int(16*muliplier), int(16*muliplier), expand_rate, act=self.act)
        self.conv24 = IdentifyBlock(int(16*muliplier), int(16*muliplier), expand_rate, act=self.act)
        self.conv25 = IdentifyBlock(int(16*muliplier), int(16*muliplier), expand_rate, act=self.act)

        initialize_weights(self)

    def forward(self, x):

        x = self.conv11(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)

        return x

class MainStreamModule(nn.Module):
    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=None):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv11 = IdentifyBlock(int(16*muliplier), int(32*muliplier), expand_rate, stride=2, act=self.act)

        self.conv21 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv22 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv23 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv24 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv25 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)

        # Pyramic scale
        self.convp1 = IdentifyBlock(int(32*muliplier), int(64*muliplier), expand_rate, stride=1, act=self.act)
        self.convp2 = Conv(int(64*muliplier), int(64*muliplier), k=3, s=2, act=nn.PReLU())
        self.convp3 = Conv(int(64*muliplier), int(64*muliplier), k=5, act=nn.PReLU())

        self.concat = Concat()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        initialize_weights(self)

    def forward(self, x):

        x = self.conv11(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)

        x1 = self.convp1(x)
        x1f = self.global_pool(x1).flatten(start_dim=1)

        x2 = self.convp2(x1)
        x2f = self.global_pool(x2).flatten(start_dim=1)

        x3 = self.convp3(x2)
        x3f = self.global_pool(x3).flatten(start_dim=1)
        
        x = self.concat([x1f, x2f, x3f])

        return x

class FERegress(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(in_ch, int(out_ch*2)),
            nn.Tanh(),
            nn.Linear(int(out_ch*2), out_ch),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.stack(x)
        return x

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def is_fused(self):
        return False

class LightWeightModel(BaseModel):

    def __init__(self, imgz=128, muliplier=1):
        super().__init__()

        # backbone, output size = size/4
        self.backbone = LightWeightBackbone(muliplier=muliplier)

        # mainstream, output size:
        # x1 = size/8
        # x2 = size/16
        # x3 = size/16
        self.mainstream = MainStreamModule(muliplier=muliplier)

        output_ch = int(64 + 64 + 64)
        
        self.logit = FERegress(output_ch, 70*2)

        initialize_weights(self)

    def forward(self, x):

        x = self.backbone(x)
        
        x = self.mainstream(x)
        
        x = self.logit(x)

        return x

class LandmarkModel(BaseModel):

    def __init__(self, depth_muliplier=1):
        super().__init__()
        
        self.stack1 = nn.Sequential(
            nn.Conv2d( 3, 32, 3, stride=2, padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU()
        )

        self.stack2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU()
        )

        self.stack3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.stack4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.stack5 = nn.Sequential(
            nn.Linear(256, 192),
            nn.Tanh(),
            nn.Linear(192, 70*2),
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
    