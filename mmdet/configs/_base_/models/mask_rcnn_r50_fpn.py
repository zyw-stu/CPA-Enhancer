# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import RoIAlign, nms
from mmengine.model.weight_init import PretrainedInit
from torch.nn import BatchNorm2d

from mmdet.models.backbones.resnet import ResNet
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.necks.fpn import FPN
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import \
    Shared2FCBBoxHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.task_modules.prior_generators.anchor_generator import \
    AnchorGenerator
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler

# model settings
model = dict(
    # 检测器名
    type=MaskRCNN,
    # 数据预处理配置
    data_preprocessor=dict(
        type=DetDataPreprocessor, # 数据预处理类型
        mean=[123.675, 116.28, 103.53], # 用于预训练骨干网络的图像归一化通道均值，按 R、G、B 排序
        std=[58.395, 57.12, 57.375], # 用于预训练骨干网络的图像归一化通道标准差，按 R、G、B 排序
        bgr_to_rgb=True, # 是否将图片通道从 BGR 转为 RGB
        pad_mask=True, # 是否填充实例分割掩码
        pad_size_divisor=32),  # padding 后的图像的大小应该可以被 ``pad_size_divisor`` 整除
    # 主干网络的配置文件
    backbone=dict(
        type=ResNet, # 主干网络的类别
        depth=50, # 主干网络的深度
        num_stages=4, # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入
        out_indices=(0, 1, 2, 3), # 每个状态产生的特征图输出的索引
        frozen_stages=1, # 第一个状态的权重被冻结
        norm_cfg=dict( # 归一化层(norm layer)的配置项
            type=BatchNorm2d, # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True), # 是否训练归一化里的 gamma 和 beta
        norm_eval=True, # 是否冻结 BN 里的统计项
        style='pytorch', # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积
        init_cfg=dict(
            type=PretrainedInit, checkpoint='torchvision://resnet50')), # 加载通过 ImageNet 预训练的模型
    neck=dict(
        type=FPN, # 检测器的 neck 是 FPN
        in_channels=[256, 512, 1024, 2048], # 输入通道数，这与主干网络的输出通道一致
        out_channels=256, # 金字塔特征图每一层的输出通道
        num_outs=5), # 输出的范围(scales)
    rpn_head=dict(
        type=RPNHead, # rpn_head 的类型是 'RPNHead'
        in_channels=256, # 每个输入特征图的输入通道，这与 neck 的输出通道一致
        feat_channels=256,  # head 卷积层的特征通道
        anchor_generator=dict( # 锚点(Anchor)生成器的配置
            type=AnchorGenerator, # 大多数方法使用 AnchorGenerator 作为锚点生成器, SSD 检测器使用 `SSDAnchorGenerator`
            scales=[8], # 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0], # 高度和宽度之间的比率
            strides=[4, 8, 16, 32, 64]), # 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes
        bbox_coder=dict( # 在训练和测试期间对框进行编码和解码
            type=DeltaXYWHBBoxCoder,# 框编码器的类别
            target_means=[.0, .0, .0, .0], # 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]), # 用于编码和解码框的标准差
        loss_cls=dict( # 分类分支的损失函数配置
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0), # 分类分支的损失类型，我们也支持 FocalLoss 等
        loss_bbox=dict( # 回归分支的损失函数配置
            type=L1Loss,  # 损失类型
            loss_weight=1.0)), # 回归分支的损失权重
    roi_head=dict( # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步
        type=StandardRoIHead,  # RoI head 的类型
        bbox_roi_extractor=dict( # 用于 bbox 回归的 RoI 特征提取器
            type=SingleRoIExtractor, # RoI 特征提取器的类型
            roi_layer=dict( # RoI 层的配置
                type=RoIAlign, # RoI 层的类别
                output_size=7, # 特征图的输出大小
                sampling_ratio=0), # 提取 RoI 特征时的采样率。0 表示自适应比率
            out_channels=256, # 提取特征的输出通道
            featmap_strides=[4, 8, 16, 32]), # 多尺度特征图的步幅，应该与主干的架构保持一致
        bbox_head=dict( # RoIHead 中 box head 的配置
            type=Shared2FCBBoxHead, # bbox head 的类别
            in_channels=256, # bbox head 的输入通道。
            fc_out_channels=1024, # FC 层的输出特征通道
            roi_feat_size=7, # 候选区域(Region of Interest)特征的大小
            num_classes=80, # 分类的类别数量
            bbox_coder=dict( # 第二阶段使用的框编码器
                type=DeltaXYWHBBoxCoder, # 框编码器的类别
                target_means=[0., 0., 0., 0.], # 用于编码和解码框的均值
                target_stds=[0.1, 0.1, 0.2, 0.2]), # 编码和解码的标准差。
            reg_class_agnostic=False, # 回归是否与类别无关
            loss_cls=dict( # 分类分支的损失函数配
                type=CrossEntropyLoss, # 分类分支的损失类型
                use_sigmoid=False, # 是否使用 sigmoid
                loss_weight=1.0),  # 分类分支的损失权重
            loss_bbox=dict( # 回归分支的损失函数配置
                type=L1Loss, # 损失类型
                loss_weight=1.0)), # 回归分支的损失权重
        mask_roi_extractor=dict( # 用于 mask 生成的 RoI 特征提取器
            type=SingleRoIExtractor, # RoI 特征提取器的类型
            roi_layer=dict( # 提取实例分割特征的 RoI 层配置
                type=RoIAlign, # RoI 层的类型
                output_size=14, # 特征图的输出大小
                sampling_ratio=0), # 提取 RoI 特征时的采样率
            out_channels=256, # 提取特征的输出通道
            featmap_strides=[4, 8, 16, 32]), # 多尺度特征图的步幅
        mask_head=dict( # mask 预测 head 模型
            type=FCNMaskHead, # mask head 的类型，更多细节请参考
            num_convs=4, # mask head 中的卷积层数
            in_channels=256, # 输入通道，应与 mask roi extractor 的输出通道一致
            conv_out_channels=256, # 卷积层的输出通道
            num_classes=80, # 要分割的类别数
            loss_mask=dict( # mask 分支的损失函数配置
                type=CrossEntropyLoss,  # 用于分割的损失类型
                use_mask=True, # 是否只在正确的类中训练 mask
                loss_weight=1.0))), # mask 分支的损失权重
    # rpn和rcnn训练配置
    train_cfg=dict(
        rpn=dict( # rpn 的训练配置
            assigner=dict( # 分配器(assigner)的配置
                type=MaxIoUAssigner, # 分配器的类型，MaxIoUAssigner 用于许多常见的检测器
                pos_iou_thr=0.7, # IoU >= 0.7(阈值) 被视为正样本
                neg_iou_thr=0.3, # IoU < 0.3(阈值) 被视为负样本
                min_pos_iou=0.3, # 将框作为正样本的最小 IoU 阈值
                match_low_quality=True, # 是否匹配低质量的框
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict( # 正/负采样器(sampler)的配置
                type=RandomSampler,  # 采样器类型
                num=256, # 样本数量
                pos_fraction=0.5, # 正样本占总样本的比例
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限
                add_gt_as_proposals=False), # 采样后是否添加 GT 作为 proposal
            allowed_border=-1,  # 填充有效锚点后允许的边框
            pos_weight=-1, # 训练期间正样本的权重
            debug=False), # 是否设置调试(debug)模式
        rpn_proposal=dict( # 在训练期间生成 proposals 的配置
            nms_pre=2000, # NMS 前的 box 数
            max_per_img=1000, #  # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type=nms, # NMS 的类别
                iou_threshold=0.7), # NMS 的阈值
            min_bbox_size=0), # 允许的最小 box 尺寸
        rcnn=dict( # roi head 的配置。
            assigner=dict( # 第二阶段分配器的配置，这与 rpn 中的不同
                type=MaxIoUAssigner, # 分配器的类型
                pos_iou_thr=0.5, # IoU >= 0.5(阈值)被认为是正样本
                neg_iou_thr=0.5, # IoU < 0.5(阈值)被认为是负样本
                min_pos_iou=0.5, # 将 box 作为正样本的最小 IoU 阈值
                match_low_quality=True, # 是否匹配低质量下的 box
                ignore_iof_thr=-1), # 忽略 bbox 的 IoF 阈值
            sampler=dict(
                type=RandomSampler, # 采样器的类型
                num=512, # 样本数量
                pos_fraction=0.25, # 正样本占总样本的比例
                neg_pos_ub=-1, # 基于正样本数量的负样本上限
                add_gt_as_proposals=True), # 采样后是否添加 GT 作为 proposal
            mask_size=28, # mask 的大小
            pos_weight=-1,  # 训练期间正样本的权重
            debug=False)), # 是否设置调试模式
    # 用于测试 rpn 和 rcnn 超参数的配置
    test_cfg=dict(
        rpn=dict( # 测试阶段生成 proposals 的配置
            nms_pre=1000, # NMS 前的 box 数
            max_per_img=1000, # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type=nms, # NMS 的类型
                iou_threshold=0.7), # NMS 阈值
            min_bbox_size=0), # box 允许的最小尺寸
        rcnn=dict( # roi heads 的配置
            score_thr=0.05, # bbox 的分数阈值
            nms=dict( # 第二步的 NMS 配置
                type=nms, # NMS 的类型
                iou_threshold=0.5), # NMS 的阈值
            max_per_img=100,  # 每张图像的最大检测次数
            mask_thr_binary=0.5)))# mask 预处理的阈值
