# ISO/IEC 42001:2023 Annex A: control mapping

For each of the 38 Annex A controls: what the Interactive Labelling platform
provides, and what the organisation has to do. Same legend as
[iso27001-annex-a.md](iso27001-annex-a.md): **Yes** (implemented, evidence
named), **Partial** (the app helps, the organisation completes it), **Org**
(organisational only).

**The AI system in scope.** Interactive segmentation models (RITM,
SimpleClick, SAM, SAM 2, or an ensemble of them) propose masks from a
labeller's clicks or box. A human accepts, corrects or rejects each
proposal; accepted labels become training data for downstream models. Online
adaptation (desktop app and benchmark) fine-tunes small LoRA adapters on the
user's own corrected masks, and admins can train a project model (a SAM 2
adapter) on a project's recorded clicks and saved masks in the web app; it is
scored against the base model on held-out images and only used once an admin
chooses it. The AI never labels without a human decision.

## A.2 Policies related to AI

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.2.2 | AI policy | Org | Write the AI policy: AI assists labelling and never labels alone; humans own every label. |
| A.2.3 | Alignment with other organisational policies | Org | Align with the ISMS policy and data protection policy (health images). |
| A.2.4 | Review of the AI policy | Org | Review yearly or when models or uses change. |

## A.3 Internal organisation

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.3.2 | AI roles and responsibilities | Partial | In the app, admins choose the model for everyone (logged) and labellers decide on each proposal. Name a model owner who approves new models and adaptation runs. |
| A.3.3 | Reporting of concerns | Org | Give labellers a channel to report bad or biased proposals (e.g. masks that systematically miss lesions on darker skin). |

## A.4 Resources for AI systems

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.4.2 | Resource documentation | Yes | [model-register.md](model-register.md) and [data-register.md](data-register.md). |
| A.4.3 | Data resources | Yes | [data-register.md](data-register.md): source, licence, use and split of every dataset. |
| A.4.4 | Tooling resources | Partial | Libraries in `requirements.txt` / `frontend/package.json`; SAM and SAM 2 pinned to upstream commits; benchmark tools in `tools/`. |
| A.4.5 | System and computing resources | Partial | Runs on CPU or one GPU; one model shared by all sessions. Record the production host. |
| A.4.6 | Human resources | Org | Labellers trained on the protocol and on automation bias; a model owner competent to read benchmark results. |

## A.5 Assessing impacts of AI systems

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.5.2 | AI system impact assessment process | Partial | Draft assessment in [ai-impact-assessment.md](ai-impact-assessment.md). Adopt the process: repeat it when adding a model, a task type or a new data domain. |
| A.5.3 | Documentation of AI system impact assessments | Partial | Keep the assessment under document control; sign off. |
| A.5.4 | Assessing AI system impact on individuals or groups of individuals | Partial | Covered in the draft (patients whose images are labelled; people affected by models trained on the labels; labellers). Complete with the clinical lead. |
| A.5.5 | Assessing societal impacts of AI systems | Partial | Covered in the draft (bias propagating into downstream clinical models). |

## A.6 AI system life cycle

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.6.1.2 | Objectives for responsible development of AI system | Org | Objectives: fewer clicks per accurate label without lowering label quality; no label without human acceptance. |
| A.6.1.3 | Processes for responsible AI system design and development | Partial | Changes through pull requests with tests; model changes evaluated on held-out benchmarks before use (`tools/click_benchmark.py`). |
| A.6.2.2 | AI system requirements and specification | Partial | Requirements: human-in-the-loop, provenance per label, rollback of adaptation, held-out evaluation. Record them in the project's requirements. |
| A.6.2.3 | Documentation of AI system design and development | Yes | Module docstrings (`isegm/inference/predictors/ensemble.py`, `isegm/adaptation/online_adapter.py`), README, [model-register.md](model-register.md). |
| A.6.2.4 | AI system verification and validation | Yes | Simulated-user benchmarks on held-out data: COCO128-seg people and the ISIC 2018 lesion set (`tools/build_lesion_benchmark.py`), scoring mIoU and clicks to 85% / 90% IoU; results in the model register. Adaptation is checked to roll back exactly to the baseline. Every project model is scored with the same simulated user against its base model on held-out images of its project before it can be chosen. |
| A.6.2.5 | AI system deployment | Partial | Only admins switch the production model; every switch is logged with checkpoint SHA-256 (`model.load`). Deploy only models listed in the register. |
| A.6.2.6 | AI system operation and monitoring | Partial | Every assisted annotation records the model; watch how often assisted labels are corrected or deleted (audit log `annotation.update` / `annotation.delete` record the label's source; the annotation's `model` field says which model proposed it). Re-run the benchmark when the data domain changes. |
| A.6.2.7 | AI system technical documentation | Yes | [model-register.md](model-register.md); code documentation; this pack. |
| A.6.2.8 | AI system recording of event logs | Yes | Audit log (model switches, annotation create / update / delete with source) plus per-annotation provenance (`model` field: type, checkpoint, SHA-256, ensemble members and fusion). |

## A.7 Data for AI systems

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.7.2 | Data for development and enhancement of AI system | Partial | [data-register.md](data-register.md) states what trained, tuned and evaluated each model. Adaptation uses only masks the user accepted in the current session. |
| A.7.3 | Acquisition of data | Partial | Public datasets with recorded licences; uploaded images come from the organisation. Record the legal basis for clinical images. |
| A.7.4 | Quality of data for AI systems | Partial | Per-project protocol shown with every task; labeller and model recorded on every label. No second-reader review yet: add one for data that trains clinical models. |
| A.7.5 | Data provenance | Yes | Each label stores its labeller, time, source (manual / assisted) and model with checksum; exports carry it (COCO `attributes`, CSV `model`, `model_sha256`). Benchmark set built reproducibly from a recorded source with a seeded split (`datasets/lesion_bench/manifest.json`). |
| A.7.6 | Data preparation | Partial | Preparation steps scripted (`tools/build_lesion_benchmark.py`: resizing, mask binarisation, split). Document preparation for any training set exported from the platform. |

## A.8 Information for interested parties of AI systems

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.8.2 | System documentation and information for users | Partial | The UI shows the active model and, for ensembles, each member's weight on the latest click. Tell labellers in the protocol that proposals can be wrong and how to correct them. |
| A.8.3 | External reporting | Org | Datasets released from the platform should state that labels were AI-assisted and include the provenance fields. |
| A.8.4 | Communication of incidents | Org | e.g. a model found to produce systematically wrong masks: identify affected labels by model SHA-256 and notify dataset users. |
| A.8.5 | Information for interested parties | Org | |

## A.9 Use of AI systems

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.9.2 | Processes for responsible use of AI systems | Partial | Built in: no label is saved without a human action; the labeller can undo, correct or label manually; adaptation can be rolled back. |
| A.9.3 | Objectives for responsible use of AI system | Org | |
| A.9.4 | Intended use of the AI system | Yes | Intended use stated in [model-register.md](model-register.md) and [ai-impact-assessment.md](ai-impact-assessment.md): annotation assistance under human control. Not for diagnosis. |

## A.10 Third-party and customer relationships

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| A.10.2 | Allocating responsibilities | Partial | Model suppliers and their licences in [model-register.md](model-register.md). The organisation is responsible for validating models on its own data. |
| A.10.3 | Suppliers | Partial | Third-party checkpoints are fingerprinted (SHA-256) and loaded without code execution; one is a third-party mirror (COCO RITM), flagged in the register. |
| A.10.4 | Customers | Org | If labelled datasets are delivered to others, state AI assistance and provenance. |
