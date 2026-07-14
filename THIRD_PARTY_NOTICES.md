# Third-party notices

This project incorporates source code from the following upstream projects.
Their original copyright and licence notices are retained in the affected
files.

## RITM (Reviving Iterative Training with Mask Guidance)

- Source: `https://github.com/SamsungLabs/ritm_interactive_segmentation`
  (also published as `saic-vul/ritm_interactive_segmentation`; both
  repositories were no longer reachable on GitHub as of July 2026)
- Licence: MIT, Copyright (c) 2021 Samsung Electronics Co., Ltd.
- Full text: see [LICENSE](LICENSE)

This repository is derived from RITM. The `isegm/` package, the training and
evaluation scripts, and the `models/` experiment configs originate there.

## SimpleClick (Interactive Image Segmentation with Simple Vision Transformers)

- Source: `https://github.com/uncbiag/SimpleClick` (branch `v1.0`)
- Licence: MIT

Vendored, adapted from the above:

- `isegm/model/is_plainvit_model.py`
- `isegm/model/modeling/models_vit.py`
- `isegm/model/modeling/pos_embed.py`
- `isegm/model/modeling/seg_head.py` (re-implemented without the mmcv
  dependency; module and parameter names follow the original so that
  SimpleClick checkpoints load unmodified)

## OpenMMLab (mmsegmentation / mmcv)

- Source: `https://github.com/open-mmlab/mmsegmentation`
- Licence: Apache License 2.0

Vendored under `isegm/model/modeling/transformer_helper/` and
`isegm/model/modeling/swin_transformer_helper/`. These files retain their
original `# Copyright (c) OpenMMLab. All rights reserved.` headers.

## Segment Anything (SAM) and SAM 2

Used as installed dependencies (`segment_anything`, `sam2`), not vendored. Both
are published by Meta under the Apache License 2.0. Model checkpoints are not
distributed with this repository and must be obtained from their upstream model
zoos.
