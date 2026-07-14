import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel

try:
    from .modeling.swin_transformer import SwinTransformer, SwinTransfomerSegHead
    _SWIN_AVAILABLE = True
except ImportError as _swin_err:
    _SWIN_AVAILABLE = False
    _swin_err_msg = str(_swin_err)


class SwinformerModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        head_params={},
        **kwargs
    ):
        if not _SWIN_AVAILABLE:
            raise ImportError(
                f"SwinformerModel requires timm and mmcv 1.x.\n"
                f"Original error: {_swin_err_msg}\n"
                "Install compatible versions: pip install timm mmcv==1.7.2"
            )
        super().__init__(**kwargs)

        # SwinTransformer receives raw coord maps; override maps_transform.
        self.maps_transform = nn.Identity()

        self.backbone = SwinTransformer(**backbone_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.backbone(image, coord_features)
        return {'instances': self.head(backbone_features), 'instances_aux': None}
