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

        self.c2f1 = C2f(int(8*muliplier), int(16*muliplier), act=self.act)
        self.c2f2 = C2f(int(16*muliplier), int(32*muliplier), act=self.act)

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

        self.conv11 = Conv(int(32*muliplier), int(32*muliplier), k=3, s=2, act=self.act)
        self.conv12 = C2f(int(32*muliplier), int(48*muliplier), act=self.act)

        self.conv21 = Conv(int(48*muliplier), int(48*muliplier), k=3, s=2, act=self.act)
        self.conv22 = C2f(int(48*muliplier), int(64*muliplier), act=self.act)

        self.conv31 = Conv(int(64*muliplier), int(64*muliplier), k=3, s=2, act=self.act)
        self.conv32 = C2f(int(64*muliplier), int(92*muliplier), act=self.act)

        self.conv41 = Conv(int(92*muliplier), int(92*muliplier), k=3, s=2, act=self.act)
        self.conv42 = C2f(int(92*muliplier), int(128*muliplier), act=self.act)

        # self.convp31 = Conv(int(128*muliplier), int(128*muliplier), k=3, s=2, act=self.act)
        # self.convp32 = C2f(int(96*muliplier), int(96*muliplier), act=self.act)

        # self.concat = Concat()
        # self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # input: size / 4, 96/4 = 24

        x = self.conv11(x)
        # size / 8
        x = self.conv12(x)

        x = self.conv21(x)
        # size / 16
        x = self.conv22(x)

        x = self.conv31(x)
        # size / 32
        x = self.conv32(x)
        
        x = self.conv41(x)
        # size / 32
        x = self.conv42(x)

        x = x.flatten(start_dim=1)

        # x = self.concat([x1f, x2f, x3f])

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
        