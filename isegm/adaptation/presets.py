"""
Per-model LoRA injection presets.

Each preset returns a ``LoRAConfig`` describing which submodules to wrap
for a given model type. The image-encoder side of foundation models stays
frozen; LoRA targets the decoder/head where domain shift bites hardest.
"""

from __future__ import annotations

from .lora import LoRAConfig, inject_lora


# Default LoRA ranks per model family (user-confirmed: r=4 for RITM/SimpleClick,
# r=8 for SAM/SAM2 mask decoder).
DEFAULT_RANKS = {
    'ritm':        4,
    'simpleclick': 4,
    'sam':         8,
    'sam2':        8,
}


def _ritm_config(rank: int) -> LoRAConfig:
    """RITM (HRNetModel): freeze the heavy HRNet backbone, target the OCR
    head + click-fusion + classifier conv layers."""
    return LoRAConfig(
        rank=rank,
        alpha=2.0 * rank,
        target_substrings=(
            'ocr_distri_head',
            'ocr_gather_head',
            'conv3x3_ocr',
            'cls_head',
            'aux_head',
            'maps_transform',
            'rgb_conv',
        ),
        # Keep the HRNet trunk untouched — it's the heaviest component
        # and shouldn't be domain-adapted with a tiny support set.
        exclude_substrings=(
            'feature_extractor.stage',
            'feature_extractor.conv1',
            'feature_extractor.conv2',
            'feature_extractor.bn',
            'feature_extractor.transition',
            'feature_extractor.layer1',
        ),
        include_conv2d=True,
    )


def _simpleclick_config(rank: int) -> LoRAConfig:
    """SimpleClick (PlainVitModel): freeze ViT backbone, target neck + head."""
    return LoRAConfig(
        rank=rank,
        alpha=2.0 * rank,
        target_substrings=(
            'neck',                 # SimpleFPN
            'head',                 # SwinTransfomerSegHead
            'patch_embed_coords',   # click-fusion projection
        ),
        exclude_substrings=(
            'backbone.',            # full ViT trunk frozen
        ),
        include_conv2d=True,
    )


def _sam_config(rank: int) -> LoRAConfig:
    """SAM: freeze image encoder, target mask decoder + prompt encoder."""
    return LoRAConfig(
        rank=rank,
        alpha=2.0 * rank,
        target_substrings=(
            'mask_decoder',
            'prompt_encoder',
        ),
        exclude_substrings=(
            'image_encoder',
        ),
        include_conv2d=True,
    )


def _sam2_config(rank: int) -> LoRAConfig:
    """SAM2: freeze image encoder + memory components, target mask decoder."""
    return LoRAConfig(
        rank=rank,
        alpha=2.0 * rank,
        target_substrings=(
            'sam_mask_decoder',
            'sam_prompt_encoder',
        ),
        exclude_substrings=(
            'image_encoder',
            'memory_encoder',
            'memory_attention',
        ),
        include_conv2d=True,
    )


_BUILDERS = {
    'ritm':        _ritm_config,
    'simpleclick': _simpleclick_config,
    'sam':         _sam_config,
    'sam2':        _sam2_config,
}


def build_config(model_type: str, rank: int | None = None) -> LoRAConfig:
    if model_type not in _BUILDERS:
        raise ValueError(
            f"No LoRA preset for model_type={model_type!r}. "
            f"Known: {sorted(_BUILDERS)}"
        )
    if rank is None:
        rank = DEFAULT_RANKS[model_type]
    return _BUILDERS[model_type](rank)


def attach_adapter(model, model_type: str, rank: int | None = None) -> tuple[LoRAConfig, list[str]]:
    """
    One-call entry: build the preset config, inject LoRA, return (config, names).

    For SAM/SAM2 the underlying ``nn.Module`` (the one returned by
    ``sam_model_registry`` or ``build_sam2``) is what gets wrapped — those
    are the same objects the predictors hold internally.
    """
    config = build_config(model_type, rank)
    wrapped = inject_lora(model, config)
    return config, wrapped
