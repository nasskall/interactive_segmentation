"""Lazily-exposed helpers ported from mmsegmentation.

Several modules in this package (``decode_head``, ``embed``, ``logger``) are
still written against the mmcv 1.x API (``mmcv.runner``, ``mmcv.utils``), which
no longer exists in mmcv 2.x. Importing them eagerly here made the whole
package unimportable under mmcv 2.x -- which in turn broke *unpickling* a
SimpleClick checkpoint, because its saved config references
``transformer_helper.cross_entropy_loss.CrossEntropyLoss`` and Python runs this
``__init__`` before it can reach that submodule.

Resolving the names on demand keeps the SimpleClick path (which only needs
``cross_entropy_loss``) working, while the mmcv 1.x-only modules still raise
their original ImportError if something actually asks for them.
"""

import importlib

_EXPORTS = {
    'PatchEmbed': '.embed',
    'nchw_to_nlc': '.shape_convert',
    'nlc_to_nchw': '.shape_convert',
    'resize': '.wrappers',
    'Upsample': '.wrappers',
    'get_root_logger': '.logger',
    'BaseDecodeHead': '.decode_head',
    'BACKBONES': '.builder',
    'HEADS': '.builder',
    'LOSSES': '.builder',
    'SEGMENTORS': '.builder',
    'build_backbone': '.builder',
    'build_head': '.builder',
    'build_loss': '.builder',
    'build_segmentor': '.builder',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
