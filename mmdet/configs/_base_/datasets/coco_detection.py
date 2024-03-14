# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset  # 数据集类型，这将被用来定义数据集
data_root = 'data/coco/'  # 数据的根路径

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# 训练数据处理流程
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args), # 第 1 个流程，从文件路径里加载图像。
    dict(type=LoadAnnotations, # 第 2 个流程，对于当前图像，加载它的注释信息
         with_bbox=True), # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
    dict(type=Resize, # 变化图像和其标注大小的流程。
         scale=(1333, 800), # 图像的最大尺寸
         keep_ratio=True),  # 是否保持图像的长宽比
    dict(type=RandomFlip, # 翻转图像和其标注的数据增广流程。
         prob=0.5), # 翻转图像的概率
    dict(type=PackDetInputs) # 将数据转换为检测器输入格式的流程
]
# 测试数据处理流程
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args), # 第 1 个流程，从文件路径里加载图像。
    dict(type=Resize, scale=(1333, 800), keep_ratio=True), # 变化图像大小的流程。
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True), # 加载标注
    dict(
        type=PackDetInputs, # 将数据转换为检测器输入格式的流程
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# 训练 dataloader 配置
train_dataloader = dict(
    batch_size=2, # 单个 GPU 的 batch size
    num_workers=2, # 单个 GPU 分配的数据加载线程数
    persistent_workers=True, # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict( # 训练数据的采样器
        type=DefaultSampler,  # 默认的采样器，同时支持分布式和非分布式训练
        shuffle=True),# 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type=AspectRatioBatchSampler), # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict( # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json', # 标注文件路径
        data_prefix=dict(img='train2017/'), # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32), # 图片和标注的过滤配置
        pipeline=train_pipeline,  # 这是由之前创建的 train_pipeline 定义的数据处理流程。
        backend_args=backend_args))
# 验证 dataloader 配置
val_dataloader = dict(
    batch_size=1, # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=2,# 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False, # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(type=DefaultSampler,
                 shuffle=False), # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True, # 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline,
        backend_args=backend_args))
# 测试 dataloader 配置
test_dataloader = val_dataloader

# 验证过程使用的评测器
val_evaluator = dict(
    type=CocoMetric, # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json', # 标注文件路径
    metric='bbox', # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator # 测试过程使用的评测器

# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type=CocoMetric,
#     metric='bbox',
#     format_only=True, # 只将模型输出转换为 coco 的 JSON 格式并保存
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀
