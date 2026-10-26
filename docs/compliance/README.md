# ISO/IEC 27001 and ISO/IEC 42001: compliance pack

This folder supports an organisation that runs the Interactive Labelling
platform (the FastAPI server in `server/` and the React app in `frontend/`)
inside an information security management system (ISMS, ISO/IEC 27001:2022)
and an AI management system (AIMS, ISO/IEC 42001:2023).

## What "compliant" means here

Both standards certify an **organisation's management system**, not a
piece of software. Certification needs things no code can provide: a scope
statement, an information security and AI policy, risk assessment and risk
treatment, a Statement of Applicability, competence and awareness records,
internal audits and management reviews.

What the application can do, and now does, is:

1. **Implement the technical controls** an auditor will look for in the
   system itself (access control, logging, provenance, secure transport).
2. **Produce evidence** for those controls (audit log, provenance in every
   export, model register with checksums).
3. **Document** how each Annex A control is met, split between what the
   application provides and what the organisation still has to do.

The documents assume the agreed deployment: **a lab network with a small
team of labellers**, one server, no internet exposure.

## Contents

| Document | Purpose |
|---|---|
| [iso27001-annex-a.md](iso27001-annex-a.md) | All 93 ISO/IEC 27001:2022 Annex A controls: what the app provides, what the organisation must do. Input to the Statement of Applicability. |
| [iso42001-annex-a.md](iso42001-annex-a.md) | All 38 ISO/IEC 42001:2023 Annex A controls, same split. |
| [model-register.md](model-register.md) | Every AI model the platform can run: source, licence, SHA-256, intended use, measured performance, known limitations. |
| [data-register.md](data-register.md) | Datasets used to build, tune and evaluate the models, and the data the platform stores. |
| [ai-impact-assessment.md](ai-impact-assessment.md) | Draft AI system impact assessment (ISO/IEC 42001 A.5) for assisted labelling. |
| [secure-operation.md](secure-operation.md) | How to deploy and run the platform so the controls hold: HTTPS, accounts, backups, log review, model changes, patching. |

## Controls built into the application

| Control | Where | Evidence |
|---|---|---|
| Sign-in on every endpoint; role-based access (admin / labeller); labellers see only assigned items | `server/labelling/api.py`, `server/app.py` | `tests/test_labelling.py`, `tests/test_security.py` |
| Segmentation sessions private to the account that opened them; only admins switch the shared model | `server/app.py` | `tests/test_security.py` |
| Passwords hashed with scrypt; 12-character minimum; bearer tokens stored only as SHA-256 and expiring after 12 hours; revoked on password change and deactivation | `server/labelling/auth.py` | `tests/test_labelling.py` |
| Sign-in throttling: 5 failures in 15 minutes lock the username and the client address for 15 minutes | `server/labelling/auth.py` | `tests/test_security.py` |
| Tamper-evident audit log of sign-ins, failures, lockouts, account and role changes, project / class / item / annotation changes, exports and model switches; hash-chained, with an integrity check | `server/labelling/audit.py`, admin page **Audit log** | `GET /api/audit`, `GET /api/audit/verify` |
| AI provenance: every assisted annotation records the model type, checkpoint and its SHA-256 (all members for an ensemble; the adapter's id and SHA-256 for a project model), the clicks that produced it and the labeller who accepted it; COCO and CSV exports carry the model and labeller | `server/sessions.py` (`provenance`), `server/labelling/export.py` | `tests/test_security.py`, `tests/test_project_models.py` |
| Project models: training on a project's own labels with held-out images, an in-app register (base and adapter SHA-256, held-out scores), explicit admin choice before use, SHA-256 check before loading, audit-logged; automatic models only propose labels, each accepted or rejected by a labeller | `server/labelling/training.py`, `isegm/adaptation/project_training.py`, `isegm/adaptation/promptless.py`, project page **Models** tab, workspace **Auto-label** | `tests/test_project_models.py`, `tests/test_auto_models.py`, `tests/test_project_training.py`, `tests/test_promptless.py` |
| Security headers: Content-Security-Policy, frame denial, `nosniff`, no-referrer, `Cache-Control: no-store` on the API, HSTS over HTTPS | `server/app.py` | `tests/test_security.py` |
| HTTPS (`--ssl-certfile` / `--ssl-keyfile`); a warning when serving beyond localhost over plain HTTP | `server/main.py` | startup log |
| Upload size limits, image decoding checks, path-traversal-safe checkpoint selection, checkpoints and adapters loaded with `weights_only=True` (RITM / SimpleClick / adapters here; SAM and SAM 2 through their upstream loaders, which default to it on PyTorch 2.6+, and 2.14 is installed) | `server/app.py`, `server/labelling/storage.py`, `server/sessions.py`, `isegm/inference/utils.py`, `interactive_demo/controller.py` | code review |
| Reproducible model evaluation on held-out data | `tools/click_benchmark.py`, `tools/build_lesion_benchmark.py` | [model-register.md](model-register.md) |

## Known gaps (organisation or future work)

- **Encryption at rest.** The SQLite database and uploaded images are plain
  files. Put `data/` on an encrypted volume (BitLocker / LUKS), which also
  covers backups made from it. See [secure-operation.md](secure-operation.md).
- **Multi-factor authentication** is not built in. On a lab network,
  network-level controls (VPN, Wi-Fi with 802.1X) and the throttling
  above are the compensating controls; add MFA or SSO before any internet
  exposure.
- **Log retention and off-host copies.** The audit log lives in the same
  database it protects. Its hash chain detects edits, but an attacker with
  full write access could rebuild the chain. Copy the log off the host
  regularly (see [secure-operation.md](secure-operation.md)) so a rebuilt
  chain would no longer match the copies.
- **Personal data.** Dermoscopy and clinical images can be health data
  (special-category personal data under GDPR). The platform does not
  de-identify images; upload only de-identified images, or treat the server
  as a system processing health data, with a DPIA.
- **Review workflow.** There is no second-labeller review or approval
  step yet. For data that trains clinical models, add one (ISO/IEC 42001
  A.7.4 data quality).
