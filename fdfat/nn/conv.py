import math

import torch
from torch import nn

__all__ = ['autopad', 'Conv', 'DWConv', 'Concat', 'IdentifyBlock', 'initialize_weights']

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-5
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # initialize_weights(self)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class IdentifyBlock(nn.Module):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, in_ch, out_ch, expand, k=3, act=True, stride=1, dilation_rate=1):
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

        # initialize_weights(self)

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