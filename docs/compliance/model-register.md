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

**Project models** are the exception: SAM 2 adapters an admin trains in the
web app (project page, **Models** tab) on that project's recorded clicks and
saved masks. They are registered in the application itself, in the
`project_models` table, which this register delegates to: base checkpoint
and its SHA-256, adapter file and its SHA-256, training objects, the
held-out images, and the held-out scores of the base model and the trained
one (same simulated user and measures as below, on that project's own
labels). A fifth of the project's labelled images are held out and never
trained on. An admin chooses one finished model per project for assisted
labelling, after comparing those scores; the server refuses to load an
adapter whose file no longer matches its registered SHA-256, and every
annotation it proposes records the adapter's id and SHA-256 next to the
base checkpoint's. Training, cancelling, deleting and choosing a model are
audit-logged (`model.train*`, `model.delete`, `project.update`).
Held-out sets are small for young projects; prefer a model only when its
gain holds up as the project grows.

Project models come in two kinds. **Interactive** ones take clicks and are
scored against their base model as above. **Automatic** ones
(`isegm/adaptation/promptless.py`) segment every mask / polygon class of the
project with no clicks, from learned class prompts; they are trained on the
images marked done (taken as completely labelled) and scored by per-class
IoU on the held-out ones. An automatic model chosen for a project only
*proposes* labels when a labeller asks (**Auto-label** in the workspace);
each proposal becomes an annotation only when the labeller accepts it, and
the annotation's provenance records the adapter and `"mode": "automatic"`.
Its intended use is the same as above: never labels without a human
decision.

**Students** (automatic, `arch` "unet", `isegm/adaptation/student.py`) are
small standalone networks: a U-Net (`segmentation_models_pytorch`, MIT)
with a ResNet-18 encoder whose ImageNet weights come from the Hugging Face
repository `smp-hub/resnet18.imagenet`, which states no licence of its own:
its model card names it a copy of torchvision's `resnet18-5c106cde.pth`
(torchvision is BSD-3-Clause; the weights were trained on ImageNet, whose
terms limit use to non-commercial research, so check before commercial
use). The revision is pinned by `segmentation_models_pytorch` 0.5.0
(`3f2325ff978283d47aa6a1d6878ca20565622683`); `model.safetensors` SHA-256
`b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19`. Each
student's register entry records `repo@revision` and that SHA-256. A student is taught by a
finished automatic SAM 2 model of its project (`teacher_id`): it learns the
labels of the done training images and the teacher's maps on the project's
other images; held-out images are used for neither, and student and teacher
are scored on them side by side. Students propose labels like the SAM 2
kind. Any finished project model can be **downloaded** by an admin (weights,
model card with classes, training data, held-out scores and hashes,
README); a student's download also holds `model.onnx` (opset 18, the same
weights with input normalisation and sigmoid built in), whose SHA-256 is in
the model card. Downloads are audit-logged (`model.download`). A downloaded model
is a copy outside the platform's controls: its model card states its
intended use, and whoever uses it is responsible for its validation there.

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

### Fine-tuned SAM 2 students (experiment, 2026-10-20)

SAM 2.1 Tiny with LoRA (rank 8) on the prompt encoder and mask decoder,
trained by `tools/distill.py train` for 2000 steps on the 1000 images of
the ISIC 2018 Task 1 **test** split (disjoint from the validation images
above). The robot's clicks are placed using the expert masks in every run.
"Distilled" learns from the adapted SimpleClick + SAM 2 teacher's soft
masks; "control" learns from the expert masks directly. Same 40 evaluation
images as above. Neither is loaded by the web server; register the adapter
file here before it is ever used for labelling.

| Model | mIoU@1 | mIoU@5 | NoC@85 | NoC@90 |
|---|---|---|---|---|
| SAM 2.1 Tiny (base) | 0.620 | 0.845 | 4.12 | 5.50 |
| Teacher (adapted ensemble) | 0.748 | 0.913 | 3.00 | 4.12 |
| Distilled student (best, step 1000) | 0.761 | 0.908 | 2.73 | 3.77 |
| Accepted-mask student (step 2000) | 0.834 | 0.922 | 2.00 | 2.92 |
| Control student (step 2000) | 0.861 | 0.935 | 1.80 | 2.52 |
| Control student, 6000-step run (best, step 5000) | **0.870** | **0.940** | **1.60** | **2.17** |

The distilled student beats its teacher but is capped by it (the teacher's
masks average 0.90 IoU against the experts). Training on expert masks is
clearly better when they exist; distillation is only worth using where no
expert masks are available. The accepted-mask student (2026-10-26) learns
from the teacher's last mask thresholded, i.e. what a labeller would have
saved from the teacher without an expert check; it recovers most of the
control's gain without expert masks.

**Reserve split (final check, 2026-10-26).** The checkpoints above were
chosen on the eval split, so their eval scores are optimistic. The 40
reserve images of the lesion benchmark were not used for any choice before
this check.

| Model | mIoU@1 | mIoU@5 | NoC@85 | NoC@90 |
|---|---|---|---|---|
| SAM 2.1 Tiny (base) | 0.596 | 0.837 | 4.20 | 5.62 |
| Distilled student | 0.720 | 0.910 | 2.85 | 4.28 |
| Accepted-mask student | 0.825 | 0.927 | 2.10 | 3.17 |
| Control student | 0.823 | 0.935 | **1.70** | 2.80 |
| Control student, 6000-step run | **0.842** | **0.940** | 1.73 | **2.70** |

All the students still clearly beat base SAM 2 on unseen images; the
ranking is unchanged. The distilled student loses the most against its eval
score (about half a click at NoC@90). Training the control three times
longer (6000 steps, checkpoint picked from six on the eval split) gains only
0.1 click at NoC@90 and 0.02 first-click IoU on the reserve split, within
noise for 40 images: most of its eval-split gain (2.52 to 2.17) came from
picking the checkpoint on that split. About 2000 steps is enough.

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
