# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
from .single_stage_my import MySingleStageDetector
from mmdet.models.restormer import BasicRestormer

@MODELS.register_module()
class MyYOLOV3(MySingleStageDetector):
    r"""Implementation of `Yolov3: An incremental improvement
    <https://arxiv.org/abs/1804.02767>`_
    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Default: None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Default: None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional):
            Model preprocessing config for processing the input data.
            it usually includes ``to_rgb``, ``pad_size_divisor``,
            ``pad_value``, ``mean`` and ``std``. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 restormer: ConfigType,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            restormer=restormer,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
