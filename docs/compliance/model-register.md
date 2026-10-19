# Model register

Every model the platform can load, with what an auditor (ISO/IEC 42001
A.4.2, A.6.2.7, A.10.3) or a model owner needs: origin, licence,
fingerprint, intended use, measured performance and known limitations.

**Rule:** only checkpoints listed here go in `model_weights/` on the
production server. When a model is added or replaced, add or update its row,
including the SHA-256 (`sha256sum model_weights/<file>`), and re-run the
benchmarks below. The server records the SHA-256 of the model behind every
assisted annotation, so labels can always be traced back to a row here.

## Intended use (all models)

Proposing segmentation masks from a labeller's clicks or box, for a human to
accept, correct or reject while producing training labels. **Not** for
diagnosis, triage or any decision about a patient, and not for labelling
without human review.

## Checkpoints

| Model | File | SHA-256 | Source | Licence | Notes |
|---|---|---|---|---|---|
| RITM HRNet-18, lesion fine-tune | `best_checkpoint_068.pth` | `daeee5d220f1faa46b737632e90fbed651dad19b7bf435ce50db2769bde99a9f` | In-house fine-tune of RITM; bundled in the repository | Code MIT (RITM). Fine-tuned on the ISIC 2018 Task 1 training split: check that split's licence terms before commercial use | Default model at startup. |
| RITM HRNet-18, COCO+LVIS | `coco_lvis_h18_itermask.pth` | `5f69cfce354d1507e3850bfc39ee7057c8dd27b6a4910d1d2dc724916b9ee32b` | **Third-party mirror** (`lucasew/ritm_annotation` release 0.0.1); the official RITM model zoo is offline | MIT (RITM) | Hash listed unchanged in the mirror's README since v0.0.8; no official hash remains to compare. Contains tensors only; loads with `weights_only=True`. |
| SimpleClick ViT-B, COCO+LVIS | `cocolvis_vit_base.pth` | `4c8472e21cdfed6d05301f14f5c6a244f21b453a59a8aee52238dc7b12223f4a` | Official SimpleClick model zoo (Google Drive, uncbiag/SimpleClick) | MIT | |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` | Meta, `segment-anything` model zoo | Apache 2.0 | Library installed from Meta's GitHub at a pinned commit. |
| SAM 2.1 Hiera-Tiny | `sam2.1_hiera_tiny.pt` | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` | Meta, `dl.fbaipublicfiles.com/segment_anything_2/092824/` | Apache 2.0 | Library installed from Meta's GitHub at a pinned commit (the PyPI `sam2` package is a third-party upload and is not used). |

**Ensembles** combine 2 to 4 of the checkpoints above (default fusion
STAPLE, no test-time augmentation). An ensemble's provenance records every
member's file and SHA-256 plus the fusion method.

**Adapters** (online adaptation, desktop app and benchmark only) are small
LoRA weight files trained on one user's accepted masks. They are not loaded
by the web server. If adapters are ever used to produce labels, register
each one here with the base checkpoint, the data it was trained on and its
benchmark result.

## Performance

Simulated-user benchmark (`tools/click_benchmark.py`): a robot clicks the
centre of the largest error, up to 10 clicks per object. **mIoU@1** is the
mean IoU after the first click; **NoC@90** is the mean number of clicks to
reach 90% IoU (10 when never reached). Lower NoC is better. "Adapted" means
3 presses of "Adapt now" after labelling the session objects, with each
model's default learning rate and steps.

### Skin lesions: ISIC 2018 Task 1 validation, 40 held-out images (2026-10-19)

Held out from the lesion RITM (trained on the 2018 training split). Built by
`tools/build_lesion_benchmark.py`; 20 session images used for adaptation.

| Model | mIoU@1 | mIoU@5 | NoC@85 | NoC@90 | Adapted NoC@90 |
|---|---|---|---|---|---|
| RITM lesion | 0.512 | 0.848 | 5.20 | 7.05 | 8.05 (worse) |
| RITM COCO+LVIS | 0.416 | 0.835 | 5.47 | 6.47 | 6.50 |
| SAM ViT-B | 0.540 | 0.839 | 4.95 | 6.03 | 5.65 |
| SAM 2.1 Tiny | 0.620 | 0.845 | 4.12 | 5.50 | **4.42** (mIoU@1 0.752) |
| SimpleClick ViT-B | 0.500 | 0.883 | 3.73 | 4.67 | **4.15** (mIoU@1 0.559) |
| Ensemble SimpleClick + SAM 2 (mean) | 0.651 | 0.872 | 3.67 | 4.60 | **4.12** (mIoU@1 0.748, NoC@85 3.00) |

### People: COCO128-seg, 12 held-out images

| Model | NoC@90 | Adapted NoC@90 |
|---|---|---|
| SimpleClick ViT-B | 3.75 | 3.67 (neutral at its tuned setting) |
| SAM 2.1 Tiny | 5.42 | 4.50 |
| SAM ViT-B | 7.33 | 5.33 |
| RITM lesion | 7.42 | 6.50 |

## Known limitations

- **Small evaluation sets** (40 lesion images, 12 COCO images): differences
  under about half a click are within noise.
- **Skin-tone coverage.** ISIC dermoscopy images come mostly from patients
  with light skin. Performance on darker skin is unmeasured; see
  [ai-impact-assessment.md](ai-impact-assessment.md).
- **Adaptation can hurt.** Adapting the lesion RITM made it worse on both
  benchmarks. Adaptation results must be benchmarked before an adapted
  model is used for production labelling, and every press can be rolled
  back.
- **Ensembles and fusion.** Before commit `18e3bc8`, STAPLE fusion
  collapsed to empty masks on two-model ensembles, so earlier ensemble
  results (worse than their best member) are invalid. With the fix,
  SimpleClick + SAM 2 matches or beats both members, adapted or not; the
  margin at NoC@90 over adapted SimpleClick (4.12 vs 4.15) is within noise.
- **Downscaling.** Sessions work on images resized to at most 2048 px;
  saved masks are scaled back to full resolution, so very fine borders lose
  detail.
