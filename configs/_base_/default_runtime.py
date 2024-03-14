default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

#vis_backends = [dict(type='LocalVisBackend'),dict(type='WandbVisBackend')]
vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_config = dict(interval=10,
                  hooks=[
                      dict(
                          type='MMDetWandbHook',
                          init_kwargs=dict(project='mmdetection',
                                           name='yolov3_d53_8xb8-ms-608-273e_voc.py',
                                           ),
                          interval=10,
                          log_checkpoint=True,
                          log_checkpoint_metadata=True,
                          num_eval_images=50)])

log_level = 'INFO'
load_from = None
resume = False
