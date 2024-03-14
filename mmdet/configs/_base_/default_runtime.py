# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

from mmdet.engine.hooks import DetVisualizationHook
from mmdet.visualization import DetLocalVisualizer

default_scope = None # 默认的注册器域名，默认从此注册器域中寻找模块

# default_hooks 是一个字典，用于配置运行时必须使用的钩子。
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook))

env_cfg = dict(
    cudnn_benchmark=False, # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程设置
        mp_start_method='fork', # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快
        opencv_num_threads=0),  # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'), # 分布式相关设置
)

vis_backends = [dict(type=LocalVisBackend)]  # 可视化后端
visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type=LogProcessor, # 日志处理器用于处理运行时日志
    window_size=50, # 日志数值的平滑窗口
    by_epoch=True) # 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。

log_level = 'INFO'  # 日志等级
load_from = None # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
