import torch
from torch import nn

from fdfat.utils.model_utils import intersect_dicts
from fdfat.utils.logger import LOGGER

from .conv import *
from . import module, modulecsp

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def freeze(self):
        pass

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

class LWModel(BaseModel):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False, act=True, face_cls=False, freeze_lmk=False):
        super().__init__()
        self.pose_rotation = pose_rotation
        self.face_cls = face_cls
        self.freeze_lmk = freeze_lmk

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
            
        output_ch = int((128)*muliplier) + int((128)*muliplier) + int((128)*muliplier)
        self.logit = module.FERegress(output_ch, 70*2)

        if self.freeze_lmk:
            self.freeze()

        if self.face_cls:
            self.classifier = nn.Sequential(
                DWConv(int(32*muliplier), int(32*muliplier), k=3, s=2, act=self.act),
                DWConv(int(32*muliplier), int(64*muliplier), k=3, s=2, act=self.act),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(int(64*muliplier), 1),
                nn.Sigmoid()
            )

        if self.pose_rotation or self.face_cls:
            self.concat = Concat()

        initialize_weights(self)

    def freeze(self):
        freeze_parameters(self.backbone)
        freeze_parameters(self.mainstream)
        if self.pose_rotation: freeze_parameters(self.aux)
        freeze_parameters(self.logit)

    def forward(self, x):

        x = self.backbone(x)
        
        if self.pose_rotation:
            aux = self.aux(x)

        if self.face_cls:
            face_cls = self.classifier(x)

        fea = self.mainstream(x)
        
        x = self.logit(fea)

        if self.face_cls:
            x = self.concat([x, face_cls])

        if self.pose_rotation:
            x = self.concat([x, aux])

        return x

class LWCSPModel(BaseModel):

    default_act = nn.ReLU6()  # default activation

    def __init__(self, imgz=128, muliplier=1, pose_rotation=False, act=True, face_cls=False, freeze_lmk=False):
        super().__init__()
        self.pose_rotation = pose_rotation
        self.face_cls = face_cls
        self.freeze_lmk = freeze_lmk

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # backbone, output size = size/4
        self.backbone = modulecsp.LightWeightBackbone(muliplier=muliplier, act=self.act)

        self.mainstream = modulecsp.MainStreamModule(muliplier=muliplier, act=self.act)

        if pose_rotation:
            self.aux = modulecsp.AuxiliaryBackbone(int(32*muliplier), 3, muliplier=muliplier, act=self.act)
            
        output_ch = int((96)*muliplier) + int((96)*muliplier) + int((96)*muliplier)

        self.logit = modulecsp.FERegress(output_ch, 70*2)

        if self.freeze_lmk:
            self.freeze()

        if self.face_cls:
            self.classifier = nn.Sequential(
                DWConv(int(32*muliplier), int(32*muliplier), k=3, s=2, act=self.act),
                DWConv(int(32*muliplier), int(64*muliplier), k=3, s=2, act=self.act),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(int(64*muliplier), 1),
                nn.Sigmoid()
            )

        if self.pose_rotation or self.face_cls:
            self.concat = Concat()

        initialize_weights(self)

    def freeze(self):
        freeze_parameters(self.backbone)
        freeze_parameters(self.mainstream)
        if self.pose_rotation: freeze_parameters(self.aux)
        freeze_parameters(self.logit)

    def forward(self, x):

        x = self.backbone(x)
        
        if self.pose_rotation:
            aux = self.aux(x)

        if self.face_cls:
            face_cls = self.classifier(x)

        fea = self.mainstream(x)
        
        x = self.logit(fea)

        if self.face_cls:
            x = self.concat([x, face_cls])

        if self.pose_rotation:
            x = self.concat([x, aux])

        return x
