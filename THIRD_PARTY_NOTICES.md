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

## MedSAM (Segment anything in medical images)

- Source: `https://github.com/bowang-lab/MedSAM` (commit `d71e8a1`)
- Licence: Apache License 2.0
- Paper: Ma J., He Y., Li F., Han L., You C., Wang B., Nature
  Communications 15:654 (2024), doi:10.1038/s41467-024-44824-z

Adapted: the image preprocessing and box-prompt decoding of
`MedSAM_Inference.py`, in the `MedSAM` class of `tools/box_benchmark.py`
(changed: weights are loaded with `torch.load(..., weights_only=True)`
instead of through the registry's checkpoint argument). The MedSAM checkpoint
(`medsam_vit_b.pth`) is downloaded separately from the authors' release and
is not distributed here.

## IMIS-Bench (Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline)

- Source: `https://github.com/uni-medical/IMIS-Bench` (commit `f4519c0`)
- Licence: none stated in the repository

No IMIS-Bench code is included here. `tools/imis_benchmark.py` imports it
from a separate, unmodified clone at run time. The IMIS-Net checkpoint is the
authors' upload at `https://huggingface.co/1Junlong/IMIS-Net` (revision
`c3d2edd`, Apache License 2.0), downloaded separately.

## RClicks (Realistic Click Simulation for Benchmarking Interactive Segmentation)

- Source: `https://github.com/emb-ai/rclicks` (commit `e20bf76`)
- Licence: MIT, Copyright (c) 2024 embodied-ai-star
- Paper: Antonov A., Moskalenko A., Shepelev D., Krapukhin A., Soshin K.,
  Konushin A., Shakhuro V., NeurIPS 2024, arXiv:2410.11722

Adapted: the click-sampling step of `isegm/inference/clicker.py`
(`Clicker._get_next_click` with a click model) in `RClicker._get_next_click`,
`tools/rclicks_clicker.py`. The clickability network itself is imported
unmodified from a separate clone by `tools/rclicks_worker.py`; its checkpoint
(`clickability_model.pth`) is downloaded separately and not distributed here.

```
MIT License

Copyright (c) 2024 embodied-ai-star

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
