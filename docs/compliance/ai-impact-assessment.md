# AI system impact assessment: assisted labelling

**Status: draft for review.** Prepared with the development team; the
clinical lead, the data protection officer and the model owner must
complete the open points and sign off (ISO/IEC 42001 A.5.2 to A.5.5).
Reassess when a model, task type or data domain is added, or yearly.

| | |
|---|---|
| System | Interactive Labelling platform, AI-assisted annotation |
| Version assessed | branch `platform-phase1`, October 2026 |
| Owner | _to be named_ |
| Reviewers | _clinical lead, DPO, model owner_ |
| Date | _at sign-off_ |

## 1. Description

Labellers annotate images (segmentation, detection, classification, change
detection). For masks, an interactive model (RITM, SimpleClick, SAM, SAM 2,
or an ensemble) turns the labeller's clicks or box into a proposed mask.
The labeller adds clicks until the mask is right, then saves it, or draws
the label by hand. Saved labels become training and evaluation data for
other models, possibly clinical ones (e.g. skin-lesion segmentation).

**Human control:** nothing is saved without the labeller's explicit action.
Every saved label records who saved it and, if assisted, which model
proposed it (type, checkpoint, SHA-256).

**Not intended for:** diagnosis, triage, or any decision about a patient;
labelling without human review.

## 2. Who is affected

| Group | How |
|---|---|
| Patients whose images are labelled | Their images are processed; their privacy is at stake (health data). |
| People affected by downstream models | Patients diagnosed or screened by models trained on these labels inherit the labels' errors and biases. |
| Labellers | Their work is shaped by the proposals (speed, attention, de-skilling); their actions are logged. |
| Dataset users / researchers | Rely on label quality and provenance statements. |

## 3. Risks and controls

| # | Risk | Likelihood / impact (before) | Controls | Residual |
|---|---|---|---|---|
| R1 | **Automation bias:** labellers accept a plausible but wrong mask (e.g. a lesion border that follows a hair or the ruler). | Likely / high for clinical training data | Labeller must act to save; masks shown as overlays over the image; protocol per project; provenance lets affected labels be found later. **Open:** second-reader review for a sample of labels; training on typical model failures. | Medium until review exists |
| R2 | **Bias across skin tones.** Lesion models and benchmarks come from ISIC, mostly light-skinned patients. Proposals may be worse on darker skin, so labels there may be worse or slower, and downstream models inherit this. | Possible / high | Human correction of every label. **Open:** measure performance by Fitzpatrick type on a representative set (e.g. Fitzpatrick17k or local data) before clinical use; report per-group results in the model register. | Unknown until measured |
| R3 | **Model change without trace:** labels in one dataset produced by different models. | Possible / medium | Only admins switch models; switches logged with SHA-256; every assisted label stores its model; exports include it. | Low |
| R4 | **Adaptation drift:** online adaptation on a few masks makes a model worse (measured for the lesion RITM). | Likely if used unchecked / medium | Adaptation is manual, benchmarked, and each press can be rolled back exactly; not available in the web server. Register any adapter before production use. Project models (web app) are scored against the base model on held-out images of the same project, registered in the app, and used only after an admin chooses one; switching back to the shared model is one click. Automatic project models only propose labels, each accepted or rejected by a labeller, and are scored per class on held-out images first. | Low |
| R5 | **Privacy breach of health images.** | Possible / high | Sign-in on every endpoint; labellers see only assigned items; sessions private; HTTPS; audit log of access changes and exports; exports admin-only. **Open:** encryption at rest; de-identification before upload; DPIA. | Medium until open points are closed |
| R6 | **Tampered or malicious model files.** | Unlikely / high | Checkpoints fingerprinted in the register; loaded with `weights_only=True` (no code execution); one third-party mirror flagged. | Low |
| R7 | **Ensemble or model choice made on weak evidence.** | Possible / low | Benchmarks on held-out data; limitations stated (small sets, ensembles not better so far). | Low |
| R8 | **Labeller monitoring concerns** (audit log records their actions and IP address). | Possible / low | Log limited to security and change events; access admin-only. **Open:** inform labellers (privacy notice). | Low |

## 4. Benefits

Fewer clicks per accurate mask (SAM 2 after adaptation needs about 1 click
fewer to reach 90% IoU on lesions than unadapted; SimpleClick about half a
click), so more labels for the same effort, and consistent provenance for
every label.

## 5. Open actions

1. Second-reader review workflow, or a sampled quality check, for clinical
   training data (R1).
2. Measure proposal quality by skin type before using labels for clinical
   models (R2).
3. Encrypt `data/` at rest; de-identify before upload; complete a DPIA (R5).
4. Privacy notice to labellers about the audit log (R8).
5. Name the system owner and model owner; sign off this assessment.
