# Copyright (c) OpenMMLab. All rights reserved.
import warnings

try:  # mmcv 1.x
    from mmcv.cnn import MODELS as MMCV_MODELS
    from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    try:
        # mmcv 2.x removed mmcv.utils.Registry, mmcv.cnn.MODELS and the
        # bricks.registry module; the registry machinery now lives in mmengine.
        from mmengine.registry import MODELS as MMCV_MODELS
        from mmengine.registry import Registry, build_from_cfg
        MMCV_ATTENTION = MMCV_MODELS
    except ImportError:
        # Neither installed. SimpleClick inference only needs the loss
        # classes to register at import time, so a minimal registry will do.
        class Registry:
            def __init__(self, name, parent=None, scope=None):
                self.name = name
                self._modules = {}

            def register_module(self, name=None, module=None, force=False):
                def register(cls):
                    self._modules[name or cls.__name__] = cls
                    return cls
                return register(module) if module is not None else register

            def get(self, key):
                return self._modules.get(key)

            def build(self, cfg, default_args=None):
                return build_from_cfg(cfg, self, default_args)

        def build_from_cfg(cfg, registry, default_args=None):
            args = dict(cfg)
            obj_type = args.pop('type')
            for k, v in (default_args or {}).items():
                args.setdefault(k, v)
            cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
            if cls is None:
                raise KeyError(f'{obj_type!r} is not registered in {registry.name}')
            return cls(**args)

        MMCV_MODELS = MMCV_ATTENTION = None


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
