import torch
from torch import nn

from .conv import *
        
class LightWeightBackbone(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=True):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # early dropout
        self.conv11 = Conv(3, int(8*muliplier), k=3, s=2, act=self.act)

        self.c2f1 = C2f(int(8*muliplier), int(16*muliplier))
        self.c2f2 = C2f(int(16*muliplier), int(32*muliplier))

    def forward(self, x):

        x = self.conv11(x)
        x = self.c2f1(x)
        x = self.c2f2(x)

        return x
    
class AuxiliaryBackbone(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, in_ch, out_ch, muliplier=1, act=True):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.stack1 = nn.Sequential(
            nn.Conv2d(in_ch, int(32*muliplier), 3, 2, autopad(3)),
            nn.BatchNorm2d(int(32*muliplier)),
            self.act,
            nn.Conv2d(int(32*muliplier), int(64*muliplier), 3, 1, autopad(3)),
            nn.BatchNorm2d(int(64*muliplier)),
            self.act,
            nn.Conv2d(int(64*muliplier), int(64*muliplier), 3, 2, autopad(3)),
            nn.BatchNorm2d(int(64*muliplier)),
            self.act,
            nn.Conv2d(int(64*muliplier), int(128*muliplier), 3, 1, autopad(3)),
            nn.BatchNorm2d(int(128*muliplier)),
            self.act,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.stack2 = nn.Sequential(
            nn.Linear(int(128*muliplier), out_ch*3),
            nn.Tanh(),
            nn.Linear(out_ch*3, out_ch),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.stack1(x)
        x = x.flatten(start_dim=1)
        x = self.stack2(x)

        return x
    
class MainStreamModule(nn.Module):
    default_act = nn.ReLU6()  # default activation

    def __init__(self, expand_rate=2, muliplier=1, act=True):
        super().__init__()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv11 = Conv(int(32*muliplier), int(64*muliplier), k=3, s=2, act=self.act)
        self.conv12 = C2f(int(64*muliplier), int(64*muliplier))

        # self.conv11 = IdentifyBlock(int(32*muliplier), int(64*muliplier), expand_rate, stride=2, act=self.act)
        # self.conv21 = IdentifyBlock(int(64*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)

        self.conv21 = Conv(int(64*muliplier), int(96*muliplier), k=3, s=2, act=self.act)
        self.conv22 = C2f(int(96*muliplier), int(96*muliplier))

        # self.conv22 = IdentifyBlock(int(64*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)
        # self.conv23 = IdentifyBlock(int(64*muliplier), int(64*muliplier), expand_rate*2, stride=1, act=self.act)

        # Pyramic scale
        self.convp11 = Conv(int(32*muliplier), int(64*muliplier), k=3, s=2, act=self.act)
        self.convp12 = C2f(int(64*muliplier), int(64*muliplier))

        self.convp21 = Conv(int(64*muliplier), int(96*muliplier), k=3, s=2, act=self.act)
        self.convp22 = C2f(int(96*muliplier), int(96*muliplier))

        self.convp31 = Conv(int(96*muliplier), int(96*muliplier), k=3, s=2, act=self.act)
        self.convp32 = C2f(int(96*muliplier), int(96*muliplier))

        # self.convp1 = IdentifyBlock(int(32*muliplier), int(64*muliplier), expand_rate, stride=2, act=self.act)
        # self.convp2 = IdentifyBlock(int(64*muliplier), int(128*muliplier), expand_rate, stride=2, act=self.act)
        # self.convp3 = IdentifyBlock(int(64*muliplier), int(128*muliplier), expand_rate, stride=2, act=self.act)

        self.concat = Concat()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # input: size / 4, 96/4 = 24
        ori = x

        x = self.conv11(x)
        # size / 8
        stage_1 = self.conv12(x)

        x = self.conv21(stage_1)
        # size / 16
        stage_2 = self.conv22(x)

        x1 = self.convp11(ori)
        # size / 8
        x1 = self.convp12(x1)
        x1f = self.global_pool(x1).flatten(start_dim=1)

        x2 = self.convp21(stage_1)
        # size / 16
        x2 = self.convp22(x2)
        x2f = self.global_pool(x2).flatten(start_dim=1)

        x3 = self.convp31(stage_2)
        # size / 32
        x3 = self.convp32(x3)
        x3f = self.global_pool(x3).flatten(start_dim=1)

        x = self.concat([x1f, x2f, x3f])

        return x
    
class FERegress(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stack(x)
        return x
        