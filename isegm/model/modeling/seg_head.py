"""
Standalone SwinTransfomerSegHead — no mmcv / timm dependency.

This is a pure-PyTorch re-implementation of the segmentation head used by
SimpleClick's PlainVitModel.  The public API (constructor kwargs, forward
signature, state-dict keys) is identical to the original so pre-trained
checkpoints load without modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal ConvModule (Conv2d + optional BN + optional activation)
# mirrors mmcv.cnn.ConvModule just enough for SwinTransfomerSegHead
# ---------------------------------------------------------------------------

class _ConvModule(nn.Module):
    """Conv2d + optional norm + optional activation.

    The submodule *names* here (``conv`` / ``bn`` / ``activate``) deliberately
    mirror mmcv.cnn.ConvModule, because that is what makes SimpleClick's
    pre-trained head weights line up. Subclassing nn.Sequential instead yields
    index-based keys (``head.convs.0.0.weight`` vs the checkpoint's
    ``head.convs.0.conv.weight``), and since load_is_model loads with
    strict=False, every head tensor would be dropped in silence and the head
    left at random init -- which reads as a model that runs but segments
    nothing.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=(norm_cfg is None))
        self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None

        self.activate = None
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activate = nn.ReLU(inplace=True)
            elif act_type == 'GELU':
                self.activate = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


# ---------------------------------------------------------------------------
# Standalone SwinTransfomerSegHead
# ---------------------------------------------------------------------------

class SwinTransfomerSegHead(nn.Module):
    """
    Multi-scale segmentation head used by PlainVitModel / SwinformerModel.

    Parameters mirror the original SimpleClick implementation so that
    pre-trained checkpoints load with strict=False without issues.

    Args:
        in_channels  (list[int]): channel counts of each FPN level.
        channels     (int):       internal channel count after 1×1 projection.
        num_classes  (int):       output channels (1 for binary segmentation).
        in_index     (list[int]): which FPN levels to use.
        upsample     (str):       'x1' | 'x2' | 'x4' — final upsampling factor.
        norm_cfg     (dict|None): passed to _ConvModule.
        act_cfg      (dict):      passed to _ConvModule.
        dropout_ratio(float):     spatial dropout before conv_seg.
        align_corners(bool):      passed to F.interpolate.
        loss_decode  : ignored at inference time (kept for compat).
    """

    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes=1,
        in_index=None,
        upsample='x1',
        interpolate_mode='bilinear',
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=None,   # ignored at inference
        input_transform='multiple_select',  # ignored, always multiple_select
        **kwargs,           # absorb any extra kwargs from checkpoint config
    ):
        super().__init__()

        if in_index is None:
            in_index = list(range(len(in_channels)))

        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.unsample = upsample
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.out_channels = {
            'x1': channels,
            'x2': channels * 2,
            'x4': channels * 4,
        }[upsample]

        num_inputs = len(in_channels)
        assert num_inputs == len(in_index)

        self.convs = nn.ModuleList([
            _ConvModule(in_channels[i], self.out_channels, 1, 1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
            for i in range(num_inputs)
        ])

        self.fusion_conv = _ConvModule(
            self.out_channels * num_inputs, self.out_channels, 1,
            norm_cfg=norm_cfg)

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels // 2, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.Conv2d(self.out_channels // 2, self.out_channels // 2, 1),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.GELU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels // 2, self.out_channels // 4, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.Conv2d(self.out_channels // 4, self.out_channels // 4, 1),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.GELU(),
        )

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg = nn.Conv2d(self.out_channels, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transform_inputs(self, inputs):
        return [inputs[i] for i in self.in_index]

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                )
            )

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.unsample == 'x2':
            out = self.up_conv1(out)
        elif self.unsample == 'x4':
            out = self.up_conv2(self.up_conv1(out))

        return self.cls_seg(out)
