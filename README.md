# Interactive Segmentation

A desktop app for click-based interactive image segmentation. You click on an
object, the model returns a mask, and you refine it with further positive and
negative clicks.

Four backends are supported behind a single UI, and any of them can be adapted
to your own imagery with LoRA, either from a small labelled set or from the
objects you segment during a session.

## Backends

| Backend | Notes |
|---|---|
| **RITM** | HRNet-based, fully convolutional, accepts any input size. |
| **SimpleClick** | Plain ViT (ViT-B/L/H). Fixed input size, handled internally. |
| **SAM** | Segment Anything, ViT-B/L/H. |
| **SAM 2** | Hiera Tiny/Small/Base+/Large. |

Pick the architecture from the *Model* panel, load a matching checkpoint, and
click. BRS and ZoomIn options apply to the RITM-family predictors and are
disabled automatically for SAM and SAM 2.

## Domain adaptation (LoRA)

Both modes train only small low-rank adapters; the base weights and the image
encoders stay frozen.

- **Offline (few-shot)**: *Fine-tune Model* trains an adapter on a labelled
  `(image, mask)` set.
- **Online**: enable recording, segment a few objects, then press *Adapt now*.
  Every adaptation can be rolled back, and adapters can be saved and reloaded.

Adapters are written as small `.pt` files under `adapters/<model_type>/`.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
python -m pip install -r requirements.txt
```

SAM and SAM 2 are optional. Install `segment_anything` and/or `sam2` only if you
intend to use those backends; the app runs without them.

## Checkpoints

**No model weights are distributed with this repository.** They are large (the
ViT checkpoints exceed GitHub's 100 MB per-file limit) and are all available
upstream. Put whatever you download into `model_weights/`.

| Backend | Where to get it |
|---|---|
| SimpleClick | [Model zoo](https://drive.google.com/drive/folders/1zVhZefCjsTBxvyxnYMVnbkrNeRCH6y9Y) from the [SimpleClick](https://github.com/uncbiag/SimpleClick) repo, e.g. `cocolvis_vit_base.pth` |
| SAM | Model-zoo links in the [segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) repo, e.g. `sam_vit_b_01ec64.pth` |
| SAM 2 | [`sam2.1_hiera_tiny.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt), or see the [sam2](https://github.com/facebookresearch/sam2) repo |
| RITM | Included: `model_weights/best_checkpoint_068.pth`. See the note below. |

### A note on the RITM checkpoint

`demo.py` loads `model_weights/best_checkpoint_068.pth` on startup. That file
**is** included here (41 MB), so the app runs from a clean clone with no extra
downloads. It is an HRNet-based RITM model fine-tuned for skin-lesion
segmentation.

It is bundled because the original RITM repositories
(`SamsungLabs/ritm_interactive_segmentation` and
`saic-vul/ritm_interactive_segmentation`) are no longer reachable on GitHub, so
the published RITM checkpoints have no official source left. If you go looking
for other RITM weights, treat any mirror as unvetted: PyTorch checkpoints are
pickle files and can execute code on load. The loader here uses
`weights_only=True`, which is a real safeguard. Leave it on.

## Run

```bash
python demo.py
```

| Action | Control |
|---|---|
| Positive click | Left click |
| Negative click | Right click |
| Zoom / pan | Scroll wheel / right drag |
| Finish object | Space |
| Partial finish | A |

## Web app (REST API + React)

The same segmentation is also available as a web app: a FastAPI backend
(`server/`) and a React frontend (`frontend/`). It currently covers the core
workflow (model selection, image upload, positive/negative clicks, undo, finish,
auto-segment, mask download). LoRA adaptation and fine-tuning are still
desktop-only.

```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart

# Build the frontend once (needs Node.js 18+)
cd frontend && npm install && npm run build && cd ..

# Serve API + built frontend at http://127.0.0.1:8000 (API docs at /docs)
python -m server.main            # --cpu, --host, --port
```

For frontend development, run `python -m server.main` and `npm run dev` in
`frontend/` side by side. Vite serves the UI with hot reload on port 5173 and
forwards `/api` to the backend: `http://127.0.0.1:8000` by default, or
whatever `ISEG_API_URL` is set to (e.g. `http://192.168.1.50:8000`) when the
backend runs on another machine.

To reach the backend from other machines, bind it to all interfaces with
`python -m server.main --host 0.0.0.0` and open `http://<server-ip>:8000`.

Main endpoints (all under `/api`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/models` | Model types, checkpoints in `model_weights/`, active model |
| POST | `/models/load` | `{model_type, checkpoint}`: swap the shared model |
| POST | `/sessions` | Upload an image (multipart `image`), returns session state |
| GET | `/sessions/{id}/image` | Image that click coordinates refer to (longest side ≤ 2048) |
| POST | `/sessions/{id}/clicks` | `{x, y, positive}` |
| POST | `/sessions/{id}/undo`, `/finish`, `/reset`, `/auto-segment` | Editing actions |
| PATCH | `/sessions/{id}/settings` | `{prob_thresh, use_zoom_in}` |
| GET | `/sessions/{id}/mask` | Label mask PNG at the original resolution |
| DELETE | `/sessions/{id}` | Drop the session |

Every editing call returns the full session state, including an RGBA overlay
as a data URL. Sessions live in memory, at most 16 at a time with the least
recently used dropped first. All sessions share one model, and requests run
one at a time.

## Tests

```bash
python -m pytest                      # unit tests
python tools/synthetic_smoke_test.py  # end-to-end, all installed backends
```

The smoke test generates synthetic image/mask pairs and drives each backend
through clicking, auto-segment, the replay buffer, online adaptation with
rollback, few-shot training, and adapter save/load. A backend whose checkpoint
is missing is reported as an explicit `SKIP` rather than passed over silently.

Note that SAM and SimpleClick saturate on the synthetic blobs, predicting close
to the whole frame, so their IoU there is not a quality signal. The smoke test
only asserts that their plumbing works; judge segmentation quality on real
imagery.

## Credits

This project stands on four pieces of published research and their authors'
released code. All credit for the models belongs to them.

**RITM** — the codebase this project is built on. The `isegm/` package, the
training and evaluation scripts and the `models/` configs all originate here.

> Sofiiuk, Petrov, Konushin. *Reviving Iterative Training with Mask Guidance for
> Interactive Segmentation*, 2021. [arXiv:2102.06583](https://arxiv.org/abs/2102.06583)
> · Original repos: `github.com/SamsungLabs/ritm_interactive_segmentation` and
> `github.com/saic-vul/ritm_interactive_segmentation` (both are no longer
> reachable on GitHub as of July 2026) · MIT, Copyright (c) 2021 Samsung
> Electronics Co., Ltd.

**SimpleClick** — the plain-ViT backend, vendored and adapted.

> Liu, Zhang, Niethammer et al. *SimpleClick: Interactive Image Segmentation with
> Simple Vision Transformers*, ICCV 2023. [arXiv:2210.11006](https://arxiv.org/abs/2210.11006)
> · [github.com/uncbiag/SimpleClick](https://github.com/uncbiag/SimpleClick) · MIT

**Segment Anything (SAM)** — used as an installed dependency.

> Kirillov, Mintun, Ravi et al. *Segment Anything*, 2023.
> [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
> · [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
> · Apache 2.0

**SAM 2** — used as an installed dependency.

> Ravi, Gabeur, Hu et al. *SAM 2: Segment Anything in Images and Videos*, 2024.
> [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
> · [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
> · Apache 2.0

**OpenMMLab** — transformer helper modules under
`isegm/model/modeling/transformer_helper/` are taken from
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (Apache 2.0) and
retain their original copyright headers. RITM's HRNet backbone comes from
[HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification).

## Licence

MIT. See [LICENSE](LICENSE).

This project is derived from RITM and vendors code from SimpleClick and
OpenMMLab. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full
attribution.
