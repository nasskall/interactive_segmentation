# ISO/IEC 27001:2022 Annex A: control mapping

For each of the 93 Annex A controls (the ISO/IEC 27002:2022 set): what the
Interactive Labelling platform provides, and what the organisation running
it has to do. Use it as input to the Statement of Applicability (SoA); the
applicability decision and justification for each control belong to the
organisation's risk treatment, not to this file.

Legend for **App**:

- **Yes**: implemented in the application; evidence named.
- **Partial**: the application helps; the organisation must complete it.
- **Org**: an organisational, people or physical control; the application
  has no part in it.

Deployment assumed: one server on a lab network, a small labelling team,
no internet exposure (see [secure-operation.md](secure-operation.md)).

## 5. Organisational controls

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| 5.1 | Policies for information security | Org | Write and approve the ISMS policy; include the labelling platform in scope. |
| 5.2 | Information security roles and responsibilities | Partial | Two roles in the app (admin, labeller). Name who holds the admin role and who owns the platform. |
| 5.3 | Segregation of duties | Partial | Labellers cannot manage users, projects, exports or models. Keep admins few; an admin should not review their own labels once a review step exists. |
| 5.4 | Management responsibilities | Org | Management requires staff to follow the labelling protocol and security rules. |
| 5.5 | Contact with authorities | Org | Keep the data protection authority contact (health data, see 5.34). |
| 5.6 | Contact with special interest groups | Org | e.g. follow PyTorch, FastAPI and Meta SAM security advisories. |
| 5.7 | Threat intelligence | Org | Track advisories for the dependencies in `requirements.txt` and `frontend/package.json`. |
| 5.8 | Information security in project management | Org | Treat new labelling projects with sensitive images as projects needing a security check (data classification, access list). |
| 5.9 | Inventory of information and other associated assets | Partial | Assets: server host, `data/labelling.db`, `data/items/`, `model_weights/`, exports. [model-register.md](model-register.md) and [data-register.md](data-register.md) cover models and datasets; add the host and backups to the asset register. |
| 5.10 | Acceptable use of information and other associated assets | Partial | Each project carries a written protocol shown to labellers with every task. Add acceptable-use rules (no copying images off the platform, exports only by admins). |
| 5.11 | Return of assets | Org | Leavers return devices; deactivate their account (see 5.18). |
| 5.12 | Classification of information | Org | Classify each project's images (e.g. public, internal, confidential health data) and record it in the project protocol. |
| 5.13 | Labelling of information | Partial | Exports are named after the project and format. Mark confidential exports per the classification scheme. |
| 5.14 | Information transfer | Partial | HTTPS between browser and server (`--ssl-certfile/--ssl-keyfile`). Exports are admin-only and audit-logged (`project.export`); define how exports may leave the lab. |
| 5.15 | Access control | Yes | Sign-in on every endpoint; admin / labeller roles; labellers see only items assigned to them; sessions private to their owner. Evidence: `tests/test_labelling.py`, `tests/test_security.py`. |
| 5.16 | Identity management | Partial | One account per person (unique usernames); accounts are deactivated, not deleted, so history stays attributable. No shared accounts: put that rule in the policy. |
| 5.17 | Authentication information | Yes | scrypt password hashes, 12-character minimum, tokens stored only as SHA-256; admins reset passwords and every session of that user ends. Hand out initial passwords through a separate channel. |
| 5.18 | Access rights | Partial | Admins grant, change and remove roles and deactivate accounts; every change is audit-logged (`user.create`, `user.update`). Review accounts every quarter (Users page plus the `user.` filter in the audit log). |
| 5.19 | Information security in supplier relationships | Org | Model and library suppliers: see [model-register.md](model-register.md). |
| 5.20 | Addressing information security within supplier agreements | Org | Only if a third party hosts or labels; then contractually. |
| 5.21 | Managing information security in the ICT supply chain | Partial | Pinned SAM / SAM 2 installs from Meta's GitHub at fixed commits; checkpoint SHA-256 values recorded; `weights_only=True` loading blocks code execution from checkpoint files. Run `pip-audit` and `npm audit` on each update. |
| 5.22 | Monitoring, review and change management of supplier services | Org | Re-check upstream licences and advisories when updating models or libraries. |
| 5.23 | Information security for use of cloud services | Org | Not applicable to the on-premises deployment; reassess if moved to a cloud. |
| 5.24 | Information security incident management planning and preparation | Org | Define who handles incidents involving the platform. The audit log is the primary forensic source. |
| 5.25 | Assessment and decision on information security events | Partial | Lockouts (`auth.login_blocked`, `auth.login_failed` with `locked`) and integrity check failures are the events to triage. |
| 5.26 | Response to information security incidents | Org | Steps: deactivate affected accounts (revokes tokens at once), preserve `data/` and the audit log copy, check the audit log integrity. |
| 5.27 | Learning from information security incidents | Org | |
| 5.28 | Collection of evidence | Partial | Audit log with hash chain and integrity check (`GET /api/audit/verify`); keep off-host copies so evidence survives host compromise. |
| 5.29 | Information security during disruption | Org | |
| 5.30 | ICT readiness for business continuity | Partial | State is two paths: `data/` and `model_weights/`. Backups of both restore the service (see [secure-operation.md](secure-operation.md)). |
| 5.31 | Legal, statutory, regulatory and contractual requirements | Partial | Model and dataset licences in [model-register.md](model-register.md) and [data-register.md](data-register.md) (some are non-commercial). |
| 5.32 | Intellectual property rights | Partial | Upstream licences in `THIRD_PARTY_NOTICES.md` and the model register. |
| 5.33 | Protection of records | Partial | Annotations keep who created them, when, and with which model; deletions and edits are audit-logged. Back up and retain per the records policy. |
| 5.34 | Privacy and protection of PII | Partial | The app stores usernames and display names only. Images may be health data: upload de-identified images or run a DPIA; restrict projects to assigned labellers (built in). |
| 5.35 | Independent review of information security | Org | Internal audit / external audit. |
| 5.36 | Compliance with policies, rules and standards for information security | Partial | Audit log supports checking that exports and account changes follow the rules. |
| 5.37 | Documented operating procedures | Partial | [secure-operation.md](secure-operation.md) is the operating procedure; adopt it into the ISMS document control. |

## 6. People controls

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| 6.1 | Screening | Org | |
| 6.2 | Terms and conditions of employment | Org | Include confidentiality of labelled data. |
| 6.3 | Information security awareness, education and training | Partial | The project protocol is shown with every task; train labellers on handling health images and on not trusting AI masks blindly ([ai-impact-assessment.md](ai-impact-assessment.md)). |
| 6.4 | Disciplinary process | Org | |
| 6.5 | Responsibilities after termination or change of employment | Partial | Deactivating an account revokes its tokens at once and keeps its labels attributable. |
| 6.6 | Confidentiality or non-disclosure agreements | Org | For labellers handling clinical images. |
| 6.7 | Remote working | Org | Remote access only through the lab VPN. |
| 6.8 | Information security event reporting | Org | Tell users whom to report suspicious activity to (e.g. an unexpected lockout). |

## 7. Physical controls

All Org: the platform runs on the organisation's hardware.

| # | Control | App | Organisation action |
|---|---|---|---|
| 7.1 | Physical security perimeters | Org | Server in a locked room or rack. |
| 7.2 | Physical entry | Org | |
| 7.3 | Securing offices, rooms and facilities | Org | |
| 7.4 | Physical security monitoring | Org | |
| 7.5 | Protecting against physical and environmental threats | Org | |
| 7.6 | Working in secure areas | Org | |
| 7.7 | Clear desk and clear screen | Org | Sessions expire after 12 hours; lock screens when away. |
| 7.8 | Equipment siting and protection | Org | |
| 7.9 | Security of assets off-premises | Org | Exports on laptops: encrypted disks only. |
| 7.10 | Storage media | Org | Encrypt backup media. |
| 7.11 | Supporting utilities | Org | UPS for the server. |
| 7.12 | Cabling security | Org | |
| 7.13 | Equipment maintenance | Org | |
| 7.14 | Secure disposal or re-use of equipment | Org | Wipe disks that held `data/`. |

## 8. Technological controls

| # | Control | App | Application support / organisation action |
|---|---|---|---|
| 8.1 | User endpoint devices | Org | Browsers used for labelling: managed, up-to-date devices. |
| 8.2 | Privileged access rights | Yes | Admin role required for user management, projects, uploads, assignment, exports, model switching and the audit log; at least one active admin is enforced. Keep the number of admins minimal and review it. |
| 8.3 | Information access restriction | Yes | Labellers see only projects and items assigned to them; sessions are private; images are served only to authorised users. |
| 8.4 | Access to source code | Org | Restrict write access to the Git repository; review changes through pull requests. |
| 8.5 | Secure authentication | Yes | scrypt hashing, 12+ character passwords, sign-in throttling (5 failures / 15 min lock the username and the client address for 15 min), tokens expire after 12 h. MFA is not built in: compensate with network access control on the lab network, and add MFA/SSO before any internet exposure. |
| 8.6 | Capacity management | Partial | Session table capped (16 live sessions); upload size limits. Monitor disk use of `data/items/`. |
| 8.7 | Protection against malware | Partial | Uploaded files are decoded as images (OpenCV) and rejected otherwise; checkpoints load with `weights_only=True`. Endpoint protection on the host is Org. |
| 8.8 | Management of technical vulnerabilities | Partial | Run `pip-audit` and `npm audit` monthly and on each update (commit `fab0230` upgraded Vite to clear npm audit advisories). Apply OS patches. |
| 8.9 | Configuration management | Partial | Configuration is the command line of `server.main` plus the environment variable `ISEG_CORS_ORIGINS`. Record the production command line in the ISMS. |
| 8.10 | Information deletion | Partial | Deleting a project or item removes its images and annotations (logged). Deleted data stays in backups until they expire; define retention. |
| 8.11 | Data masking | Org | De-identify images before upload. |
| 8.12 | Data leakage prevention | Partial | Exports admin-only and logged; no public links; session data never shared across accounts. |
| 8.13 | Information backup | Org | Back up `data/` (database + images) and `model_weights/`, encrypted; test restores. See [secure-operation.md](secure-operation.md). |
| 8.14 | Redundancy of information processing facilities | Org | Single server is acceptable for the lab use case if backups are tested. |
| 8.15 | Logging | Yes | Audit log of sign-ins, failures, lockouts, account, project, class, item and annotation changes, exports and model switches, with user, time (UTC), target, details and client address. Admin page **Audit log**; `GET /api/audit`. |
| 8.16 | Monitoring activities | Partial | Review the audit log weekly (sign-in failures, exports, role changes). Server access logs from uvicorn go to stdout; keep them. |
| 8.17 | Clock synchronisation | Org | Keep the host on NTP; audit timestamps are the host's UTC time. |
| 8.18 | Use of privileged utility programs | Org | |
| 8.19 | Installation of software on operational systems | Org | Install dependencies only from `requirements.txt` / `package-lock.json` and the pinned upstream commits. |
| 8.20 | Networks security | Org | Bind to a lab interface only; firewall the port to the labelling subnet. |
| 8.21 | Security of network services | Partial | HTTPS supported; CORS closed unless `ISEG_CORS_ORIGINS` is set. |
| 8.22 | Segregation of networks | Org | |
| 8.23 | Web filtering | Org | |
| 8.24 | Use of cryptography | Partial | TLS for transport (organisation supplies certificate); scrypt for passwords; SHA-256 for tokens, the audit chain and checkpoint fingerprints. Encryption at rest: put `data/` on an encrypted volume. |
| 8.25 | Secure development life cycle | Partial | Changes go through pull requests with automated tests (`python -m pytest`), including security tests. |
| 8.26 | Application security requirements | Yes | Requirements implemented and listed in [README.md](README.md). |
| 8.27 | Secure system architecture and engineering principles | Partial | Default-deny access checks on every route, server-side authorisation, least privilege for labellers, defence in depth (headers, throttling, logging). |
| 8.28 | Secure coding | Partial | Parameterised SQL throughout; input validation via Pydantic models; path-traversal checks on checkpoint names; no `eval` or unsafe deserialisation. |
| 8.29 | Security testing in development and acceptance | Yes | `tests/test_security.py` covers access control, session isolation, throttling, audit integrity, provenance and headers; run on every change. |
| 8.30 | Outsourced development | Org | |
| 8.31 | Separation of development, test and production environments | Partial | `--data-dir` keeps test data apart from production; never point tests or demos at the production `data/`. |
| 8.32 | Change management | Partial | Code changes through pull requests. Model changes are audit-logged (`model.load` with checkpoint SHA-256); record model additions in [model-register.md](model-register.md). |
| 8.33 | Test information | Org | Use public or synthetic images for testing (the test suite uses synthetic images; the benchmark uses public ISIC and COCO images). |
| 8.34 | Protection of information systems during audit testing | Org | |
