_base_ = 'retinanet_r50_fpn_1x_coco.py'
# 将基于 epoch 的 RetinaNet 配置更改为基于 iteration 的示例
# training schedule for 90k
train_cfg = dict(
    _delete_=True, # 忽略继承的配置文件中的值（可选）
    type='IterBasedTrainLoop', # iter-based 训练循环
    max_iters=90000, # 最大迭代次数
    val_interval=10000) # 每隔多少次进行一次验证

# learning rate policy
# 将参数调度器修改为 iter-based
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]
# 切换至 InfiniteSampler 来避免 dataloader 重启
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))
# 将模型检查点保存间隔设置为按 iter 保存
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))
# 将日志格式修改为 iter-based
log_processor = dict(by_epoch=False)
