# Data register

Datasets used to build, tune or evaluate the AI models, and the data the
platform itself stores (ISO/IEC 42001 A.4.3, A.7; ISO/IEC 27001 5.9, 5.34).
None of the datasets are committed to the repository (`datasets/` and
`data/` are git-ignored).

## Datasets used for models

| Dataset | Used for | Location | Source and licence | Personal data |
|---|---|---|---|---|
| ISIC 2018 Task 1 **training** split | Fine-tuning the lesion RITM (`best_checkpoint_068.pth`), before this repository | not on this machine | ISIC Archive challenge data; check the licence terms of the training archive before commercial use | De-identified dermoscopy images |
| ISIC 2018 Task 1 **validation** split (100 images, one expert lesion mask each) | Lesion benchmark: 20 session images (adaptation), 40 evaluation images, 40 held in reserve | `datasets/isic2018_val/raw` (as downloaded), `datasets/lesion_bench/` (prepared) | `isic-challenge-data.s3.amazonaws.com/2018/`; **CC0** (per the archive's `ATTRIBUTION.txt`) | De-identified dermoscopy images |
| COCO128-seg (128 COCO images with polygon labels) | People benchmark and adaptation-setting sweeps | `datasets/coco128-seg` | Ultralytics; COCO images from Flickr under their own licences, COCO annotations CC BY 4.0 | Photos of people |
| COCO + LVIS, SA-1B, SA-V | Training of the upstream general models (RITM COCO, SimpleClick, SAM, SAM 2) | not used here | Upstream; see each model's paper and licence | n/a |
| Synthetic lesion images | Unit tests | generated in `tests/conftest.py` | n/a | None |

### Preparation of the lesion benchmark

`tools/build_lesion_benchmark.py` (reproducible from the raw download):
images shrunk to at most 1024 px on the long side (area interpolation),
masks resized with nearest-neighbour and binarised at 127, and a seeded
split (seed 0) written to `datasets/lesion_bench/manifest.json` with each
image's original size and lesion area. Lesions cover 1.3% to 85% of the
frame (median 22%).

### Separation of training and evaluation data

- The lesion RITM was trained on the ISIC 2018 training split; the
  benchmark uses only the validation split.
- Adaptation in the benchmark trains on the session split and is scored on
  the evaluation split; the two never overlap.

## Data the platform stores

| Data | Location | Contents | Retention / protection |
|---|---|---|---|
| Accounts | `data/labelling.db` (`users`, `tokens`) | Username, display name, role, scrypt password hash; SHA-256 of active tokens | Tokens expire after 12 h; accounts are deactivated, not deleted. |
| Projects, classes, items, annotations | `data/labelling.db` | Labelling configuration, protocols, labels with labeller, source, model provenance and timestamps; for assisted labels also the clicks and box that produced each object (`prompts`) | Organisation's records retention policy. |
| Project models | `data/labelling.db` (`project_models`), `data/models/project_<id>/model_<id>/adapter.pt` | SAM 2 LoRA adapters (interactive, or automatic with learned class prompts) trained on a project's images, clicks and saved masks, with their register entry |
| Model downloads | downloaded by admins | A project model's weights and model card (classes, item counts, held-out item ids, scores, hashes); no images or labels | Every download is audit-logged; a model trained on health images is derived from them: handle at the same classification. |
| Auto-label proposals | memory only | Masks an automatic model proposed for an image, until accepted or discarded | Last 64 requests kept; dropped on restart. Requests are audit-logged (`item.auto_label`). | Derived from the project's images: same classification as them. Deleted with the model or the project. |
| Uploaded images | `data/items/<item id>/` | Images as uploaded | **May be health data.** Upload de-identified images, or treat the server as processing special-category data (DPIA, lawful basis). Encrypted volume recommended. |
| Audit log | `data/labelling.db` (`audit_log`) | Security and change events with user, time, client address | Append-only, hash-chained; copy off-host regularly. |
| Interactive sessions | memory only | Current image and clicks | Dropped on restart or after 16 newer sessions. |
| Exports | downloaded by admins | Labels (and optionally images) with provenance | Every export is audit-logged; handle per classification. |
