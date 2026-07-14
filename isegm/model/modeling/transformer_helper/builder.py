# Copyright (c) OpenMMLab. All rights reserved.
import warnings

try:  # mmcv 1.x
    from mmcv.cnn import MODELS as MMCV_MODELS
    from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    # mmcv 2.x removed mmcv.utils.Registry, mmcv.cnn.MODELS and the
    # bricks.registry module; the registry machinery now lives in mmengine.
    from mmengine.registry import MODELS as MMCV_MODELS
    from mmengine.registry import Registry, build_from_cfg
    MMCV_ATTENTION = MMCV_MODELS


PIXEL_SAMPLERS = Registry('pixel sampler')
# Explicit scopes: both children hang off the same mmengine MODELS parent, and
# a parent registry rejects two children that infer the same scope.
MODELS = Registry('models', parent=MMCV_MODELS, scope='isegm')
ATTENTION = Registry('attention', parent=MMCV_ATTENTION, scope='isegm_attention')

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS

def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
