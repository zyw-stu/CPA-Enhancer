import torch
import torch.nn as nn
import torch.nn.functional as F
from .cpa_arch import CPA_arch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class PromptRestormer(BaseModule):
    def __init__(self,c_in=3,c_out=3,dim=32):
        super(PromptRestormer, self).__init__()
        self.prompt_unet_arch = CPA_arch(c_in, c_out, dim)
    def forward(self,x):
        x_=x
        x=self.prompt_unet_arch(x)
        x=x+x_
        return x
