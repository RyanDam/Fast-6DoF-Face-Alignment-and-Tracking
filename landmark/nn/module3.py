import torch
from torch import nn

from .conv import *
        
class LightWeightBackbone(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=None):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # early dropout
        self.conv11 = Conv(3, 16, s=2, act=nn.PReLU())

        self.conv21 = IdentifyBlock(16, int(16*muliplier), expand_rate, stride=1, act=self.act)
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
    
class AuxiliaryBackbone(nn.Module):

    default_act = nn.PReLU()  # default activation

    def __init__(self, in_ch, out_ch, muliplier=1, act=None):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.stack1 = nn.Sequential(
            nn.Conv2d(in_ch, int(32*muliplier), 3, 2, autopad(3)),
            nn.BatchNorm2d(int(32*muliplier), track_running_stats=False),
            nn.PReLU(),
            nn.Conv2d(int(32*muliplier), int(64*muliplier), 3, 1, autopad(3)),
            nn.BatchNorm2d(int(64*muliplier), track_running_stats=False),
            nn.PReLU(),
            nn.Conv2d(int(64*muliplier), int(64*muliplier), 3, 2, autopad(3)),
            nn.BatchNorm2d(int(64*muliplier), track_running_stats=False),
            nn.PReLU(),
            nn.Conv2d(int(64*muliplier), int(128*muliplier), 3, 1, autopad(3)),
            nn.BatchNorm2d(int(128*muliplier), track_running_stats=False),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.stack2 = nn.Sequential(
            nn.Linear(int(128*muliplier), out_ch*3),
            nn.Tanh(),
            nn.Linear(out_ch*3, out_ch),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):

        x = self.stack1(x)
        x = x.flatten(start_dim=1)
        x = self.stack2(x)

        return x
    
class MainStreamModule(nn.Module):
    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=None):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv11 = IdentifyBlock(int(16*muliplier), int(32*muliplier), expand_rate, stride=2, act=self.act)

        self.conv21 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv22 = IdentifyBlock(int(32*muliplier), int(32*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv23 = IdentifyBlock(int(32*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv24 = IdentifyBlock(int(64*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)
        self.conv25 = IdentifyBlock(int(64*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)

        # Pyramic scale
        self.convp1 = IdentifyBlock(int(64*muliplier), int(128*muliplier), expand_rate, stride=1, act=self.act)
        self.convp2 = Conv(int(128*muliplier), int(128*muliplier), k=3, s=2, act=nn.PReLU())
        self.convp3 = Conv(int(128*muliplier), int(128*muliplier), k=3, act=nn.PReLU())

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