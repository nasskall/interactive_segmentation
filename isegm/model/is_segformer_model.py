import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.modifiers import LRMult

try:
    from .modeling.segformer import MixVisionTransformer, SegformerHead
    _SEGFORMER_AVAILABLE = True
except ImportError as _segformer_err:
    _SEGFORMER_AVAILABLE = False
    _segformer_err_msg = str(_segformer_err)


class SegformerModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params=None,
        decode_head_params=None,
        backbone_lr_mult=0.1,
        **kwargs
    ):
        if not _SEGFORMER_AVAILABLE:
            raise ImportError(
                f"SegformerModel requires mmcv 1.x but got an incompatible version.\n"
                f"Original error: {_segformer_err_msg}\n"
                "Install a compatible mmcv with: pip install mmcv==1.7.2"
            )
        super().__init__(**kwargs)

        # SegFormer receives raw coord maps directly; override maps_transform.
        self.maps_transform = nn.Identity()

        self.feature_extractor = MixVisionTransformer(**backbone_params)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        self.head = SegformerHead(**decode_head_params)

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.feature_extractor(image, coord_features)
        return {'instances': self.head(backbone_features), 'instances_aux': None}
