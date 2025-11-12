# Watheq Project Execution Plan – Academic Showcase Edition

This plan is tailored for a bachelor-level capstone team (up to six beginner engineers) to deliver a demonstrable, classroom-ready version of "Watheq" within one semester. Every section distinguishes between **essential** work that makes the project complete, **bonus** enhancements that add polish, and **overkill** efforts that are unnecessary for academic grading.

## Quick Overview

- **Suggested duration:** 10–12 weeks (adjustable to your semester calendar).
- **Work style:** Six parallel workstreams (one per teammate) but runnable sequentially by a smaller crew or even a single person with extra time.

**Core tooling:**

- Backend: Python 3.11, FastAPI, SQLite (PostgreSQL optional), SQLAlchemy.
- AI: EasyOCR, face_recognition, scikit-image (SSIM), OpenCV; PyTorch only if you pursue bonuses.
- Web frontend: React + Vite + Tailwind CSS.
- Mobile frontend: Flutter or React Native plus Firebase Cloud Messaging for push updates.
- Storage & trust: SQLite for metadata, IPFS node for document storage, Hyperledger Fabric (or Sawtooth) network for tamper-proof logging, SHA-256 hashing utilities.
- Collaboration: Notion/Google Docs, Trello/ClickUp, GitHub or GitLab.

## Role Allocation (adjust freely)

| Track                                   | Primary Owner | Backup / Alternate              |
| --------------------------------------- | ------------- | ------------------------------- |
| Project coordination & docs             | Member 1      | Owns QA & demo if short-staffed |
| Backend & API orchestration             | Member 2      | Supported by Member 5           |
| OCR & image prep                        | Member 3      | Pair with Member 2 if needed    |
| Forgery analysis (visual checks)        | Member 4      | Supported by Member 3           |
| Face verification & mobile application  | Member 5      | Supported by Member 6           |
| Blockchain/IPFS ledger & integration QA | Member 6      | Supported by Member 1           |

> If only one or two developers are available, follow the phases in order and reuse these tracks as a checklist.

## Suggested Timeline

| Week | Primary Focus                                                    |
| ---- | ---------------------------------------------------------------- |
| 1    | Kick-off and organization                                        |
| 2    | Requirements review and lightweight design                       |
| 3    | Environment setup and data preparation                           |
| 4–6  | Core feature development (OCR, verification, UI, mobile, ledger) |
| 7    | Integration, IPFS/Fabric hardening, scenario testing             |
| 8    | Bonus refinements and extra testing (optional)                   |
| 9–10 | Documentation, presentation, demo recording                      |

---

## Phase 0: Kick-off and Governance (3 days)

### Phase 0 – Must Do (Essential)

1. **Kick-off session** (in-person or online) to align on goals, constraints, and grading rubric.
2. **Create a collaboration hub** in Notion or Google Drive containing:
   - Weekly planner.
   - Requirements backlog (user stories + status).
   - Meeting notes section.
3. **Initialize the Git repository** (GitHub or GitLab):
   - Protect `main` branch.
   - Adopt a simple flow: `main` (stable) and `feature/<task-name>` branches.
4. **Set up a task board** in Trello or ClickUp with columns Backlog → In Progress → Review → Done.
5. **Establish communication channels** (WhatsApp/Discord) with a 30-minute weekly sync and a 15-minute midweek check-in.

### Phase 0 – Nice to Have (Bonus)

- Record the kickoff call (audio/video) for reference.
- Draft a weekly progress report template inside Notion.

### Phase 0 – Overkill (Skip)

- Rolling out Jira or Azure DevOps with advanced workflows.
- Producing full PMBOK-style project documentation.

---

## Phase 1: Requirements & Lightweight Design (1 week)

### Phase 1 – Must Do (Essential)

1. **Read the requirement document together** and confirm the baseline capabilities:
   - Upload document (image/PDF) and validate its contents.
   - Extract text (OCR).
   - Detect tampering in seals/signatures.
   - Match document photo with a selfie.
   - Display consolidated results in a simple dashboard (web) and a mirrored mobile view.
   - Push the original document to IPFS and retain the CID.
   - Create a SHA-256 fingerprint and record it on the consortium blockchain with status metadata.
2. **Write user stories** using the format “As a [role], I need [feature] so that [value].”
3. **Draw a one-page architecture sketch** in draw.io or Whimsical (C4 Level 1):
   - Web frontend (React) + mobile app (Flutter/React Native) atop a shared API.
   - REST API (FastAPI) orchestrating AI, IPFS, and blockchain services.
   - AI helper modules (OCR, forgery, face match).
   - SQLite operational database, Hyperledger Fabric network, and IPFS cluster.
4. **List dependencies per module:**
   - OCR → EasyOCR + OpenCV.
   - Forgery → OpenCV + scikit-image (SSIM).
   - Face verification → face_recognition.
   - Web UI → React stack.
   - Mobile app → Flutter (Dart) or React Native + local caching strategy.
   - IPFS + blockchain → go-ipfs daemon, Hyperledger Fabric test network, Fabric SDK (Python or Node.js).
5. **Document data handling rules** (folder structure, naming scheme, accepted formats, CID retention, hash ledger schema).

### Phase 1 – Nice to Have (Bonus)

- Build a quick Figma mock-up of the admin screen.
- Capture top five project risks with mitigation ideas.

### Phase 1 – Overkill (Skip)

- Designing microservices with message brokers.
- Applying enterprise frameworks like TOGAF.

---

## Phase 2: Environment & Data Preparation (1 week)

### Phase 2 – Must Do (Essential)

1. **Set up local dev environments:**
   - Install Python 3.11, Node.js LTS, Git, and Docker Desktop (needed for Fabric test network).
   - Create a virtual env: `python -m venv .venv` → activate.
   - Install essentials: `pip install fastapi uvicorn[standard] easyocr opencv-python-headless scikit-image face-recognition python-multipart sqlalchemy alembic ipfshttpclient cryptography`.
   - Install ledger helpers: `pip install fabric-sdk-py` (or Node.js Fabric SDK if you prefer a JS gateway).
2. **Install and configure distributed services:**
   - Download and initialize `go-ipfs`; verify `ipfs daemon` runs locally and exposes the HTTP API.
   - Clone Hyperledger Fabric samples; run `test-network` with two orgs and CouchDB using `./network.sh up createChannel`.
   - Generate a channel-specific chaincode stub (e.g., `watheq_cc`) for storing document hashes and metadata.
   - Document startup and shutdown commands in `infrastructure/README.md`.
3. **Scaffold the repo structure:**
   - Folders: `backend`, `frontend`, `mobile`, `ai_models`, `ledger`, `infrastructure`, `docs`, `data`.
   - Add a `README.md` explaining how to run each part and how services connect (API ↔ IPFS ↔ Fabric).
4. **Assemble a mini dataset:**
   - 10–15 document images plus a matching selfie for each.
   - 5 tampered examples (hand-edited or digitally altered).
   - Store in `data/raw` with consistent names (`doc_01_front.jpg`, `doc_01_selfie.jpg`).
5. **Write a data organizer script** (`ai_models/prepare_data.py`) to move files into:
   - `data/processed/ocr`
   - `data/processed/forgery`
   - `data/processed/face`
6. **Initialize SQLite** via `backend/init_db.py` with tables:
   - `documents(id, user_name, document_type, upload_path, ipfs_cid, created_at)`
   - `ocr_results(document_id, extracted_text, confidence)`
   - `verification_results(document_id, seal_score, signature_score, face_score, overall_status)`
   - `document_hashes(document_id, sha256, fabric_tx_id, ledger_status, recorded_at)`

### Phase 2 – Nice to Have (Bonus)

- A lightweight Docker Compose file that runs FastAPI + SQLite + frontend.
- Use DVC to version datasets.

### Phase 2 – Overkill (Skip)

- Provisioning Kubernetes clusters or Terraform stacks.
- Running managed PostgreSQL with automated backups.

---

## Phase 3: Core Feature Development (2–3 weeks)

### 3.1 OCR Service (EasyOCR Baseline)

#### 3.1 – Must Do (Essential)

1. Create `backend/services/ocr_service.py` with `extract_text(image_path)`:
   - Instantiate `easyocr.Reader(['ar', 'en'])`.
   - Return extracted text plus confidence.
2. Add `POST /api/ocr` in `backend/main.py`:
   - Accept `UploadFile`.
   - Save to `data/uploads`.
   - Call `extract_text` and persist results in SQLite.
3. Write `tests/test_ocr.py` with pytest to confirm basic extraction works.
4. Update the README with run instructions: `uvicorn backend.main:app --reload`.

#### 3.1 – Nice to Have (Bonus)

- Preprocess images (grayscale, contrast enhancement).
- Accept multi-page PDFs (convert via `pdf2image`).

#### 3.1 – Overkill (Skip)

- Training a custom TrOCR model from scratch.
- Hosting a distributed OCR service with TorchServe.

### 3.2 Visual Forgery Checks

#### 3.2 – Must Do (Essential)

1. Implement `backend/services/forgery_service.py`:
   - Use `skimage.metrics.structural_similarity` (SSIM) against a reference (if available).
   - If no reference, run edge highlights (Canny) and compute a basic score.
2. Build `POST /api/forgery`:
   - Accept original + current images when available (or a single upload).
   - Output similarity score (0–1) with a 0.75 default threshold.
3. Save scores to `verification_results` (`seal_score`, `signature_score`).
4. Add a synthetic pytest that compares an untouched vs. lightly edited image.

#### 3.2 – Nice to Have (Bonus)

- Allow manual region-of-interest selection for signature/seal zones.
- Generate a heatmap report using matplotlib.

#### 3.2 – Overkill (Skip)

- GAN-based forgery detection.
- Training Mask R-CNN or EfficientNet on large datasets.

### 3.3 Face Verification

#### 3.3 – Must Do (Essential)

1. Add `backend/services/face_service.py` using `face_recognition`:
   - Function `compare_face(document_face_path, selfie_path)` returning similarity + distance.
   - Use 0.6 as an initial acceptance threshold.
2. Provide `POST /api/face-verify` that ingests two images and returns the decision.
3. Create `tests/test_face.py` with your sample pairs.
4. Persist results alongside the document record.

#### 3.3 – Nice to Have (Bonus)

- Liveness-lite check: request two selfies with a small head movement and compare.
- Add `scripts/crop_face.py` to auto-crop document faces.

#### 3.3 – Overkill (Skip)

- Full liveness solutions with 3D sensing.
- Training ArcFace or other deep models from scratch.

### 3.4 User Interfaces (Web + Mobile)

#### 3.4 – Must Do (Essential)

1. Scaffold the **React + Vite** web console:
   - `npm create vite@latest frontend -- --template react`
   - Install Axios, React Query, Tailwind CSS, and a component kit (e.g., Headless UI).
   - Screens: admin login, submissions dashboard (with filtering), document detail (text, scores, ledger status, CID link), bulk approval for IPFS uploads.
2. Build the **Flutter (or React Native) mobile app** for citizen submissions:
   - Bootstrapped with `flutter create watheq_mobile` (or `npx react-native init`).
   - Integrate camera/file picker for document + selfie capture, offline draft queue, submission tracking (status pulled from API).
   - Implement bilingual UI (Arabic/English) switching and basic auth (email/OTP or institutional SSO mock).
3. Share design system tokens between web and mobile (colors/typography) and document the API contract used by both clients.

#### 3.4 – Nice to Have (Bonus)

- Add responsive data visualisations (verification trends) for supervisors.
- Ship a simple push notification workflow (Firebase Cloud Messaging) to alert users when ledger confirmation arrives.

#### 3.4 – Overkill (Skip)

- Full Next.js dashboard with Storybook-driven component library.
- Native iOS + Android builds with completely separate codebases.

### 3.5 Trust Services (IPFS & Blockchain)

#### 3.5 – Must Do (Essential)

1. Implement `ledger/ipfs_service.py` (or TS equivalent) that pins uploaded files to the local IPFS node and returns the CID.
2. Add `ledger/hash_service.py` with `compute_sha256(file_path)` ensuring deterministic hashing on binary mode reads.
3. Create a Fabric chaincode module (Node.js or Go) exposing `RecordDocument(hash, cid, meta)` and `GetDocument(hash)`.
4. Integrate the backend with Fabric via SDK: on submission finalize, invoke `RecordDocument` and persist the returned transaction ID in SQLite.
5. Build retry logic for ledger writes and queue failed operations for later replays (`document_hashes.ledger_status`).
6. Expose `GET /api/ledger/<document_id>` for UI/mobile to fetch CID, hash, and ledger confirmation data.

#### 3.5 – Nice to Have (Bonus)

- Implement chaincode endorsement policies requiring two org approvals before commit.
- Add IPFS cluster replication across two peers to simulate availability zones.

#### 3.5 – Overkill (Skip)

- Migrating to production-grade Fabric ordering service with HSM-backed identities.
- Integrating public blockchains (e.g., Ethereum) for notarisation.

### 3.6 End-to-End Workflow

#### 3.6 – Must Do (Essential)

1. Expose `POST /api/process-document` in `backend/main.py` to orchestrate:
   - Save upload → OCR → forgery → face verification (if selfies provided).
   - Push file to IPFS, capture CID, compute SHA-256, call Fabric chaincode, and persist tx metadata.
   - Aggregate a final JSON decision for the UI/mobile including ledger receipt and download links.
2. Create `scripts/demo_workflow.py` CLI:
   - `python scripts/demo_workflow.py --doc data/sample/doc_01.jpg --selfie data/sample/doc_01_selfie.jpg`
   - CLI prints the CID, hash, Fabric transaction ID, and verification scores.
3. Add `tests/test_full_flow.py` ensuring the API pipeline runs on sample files, mocks ledger/IPFS when offline, and asserts hash persistence.

#### 3.6 – Nice to Have (Bonus)

- Introduce a simple job queue (e.g., `rq`) for longer tasks or ledger retries.
- Log events to `logs/app.log` (or `structlog` JSON) for debugging and audit trails.

#### 3.6 – Overkill (Skip)

- RabbitMQ/Kafka messaging or full microservice decomposition.

---

## Phase 4: Testing, Documentation, and Demo Prep (2 weeks)

### Phase 4 – Must Do (Essential)

1. **Author a test plan** in `docs/test_plan.md` covering scenarios:
   - Valid document + matching selfie → accepted.
   - Tampered document → flagged.
   - Document without selfie → partial result.
   - Ledger outage during submission → queued retry → eventual consistency.
   - Retrieval of hash + CID from Fabric and IPFS for audit purposes.
2. **Run automated tests** across stacks and archive results:
   - Backend: `pytest -v` (include mocked + live ledger/IPFS cases).
   - Web: `npm run test` (component + integration) and `npm run lint`.
   - Mobile: `flutter test` (or `npx detox test` for React Native).
3. **Write a bilingual user guide** (`docs/user_manual.md`) detailing web admin actions, mobile submission steps, and ledger verification flow.
4. **Document an operations runbook** (`docs/runbook.md`) that explains starting/stopping IPFS, Fabric, handling chaincode upgrades, and rotating credentials.
5. **Prepare the presentation deck** (PowerPoint/Google Slides): problem, approach, architecture, demo screenshots, results, future work.
6. **Record a 3–5 minute demo video** showing the web admin, mobile submission, IPFS CID lookup, and Fabric transaction explorer.

### Phase 4 – Nice to Have (Bonus)

- Formal technical report (Word/LaTeX) with diagrams and appendices.
- FAQ sheet for anticipated jury questions.

### Phase 4 – Overkill (Skip)

- Load testing with Locust/JMeter.
- Monitoring stack (Prometheus + Grafana).

---

## Phase 5: Optional Enhancements (1 week, only if time remains)

### Phase 5 – Bonus Ideas

- Upgrade forgery detection with lightweight CNN fine-tuning or transformer-based OCR re-ranking.
- Add analytics dashboards (verification throughput, tamper rate heatmaps) with scheduled exports to PDF/CSV.
- Automate Fabric chaincode CI/CD (GitHub Actions + `network.sh` scripts) and add health probes for IPFS peers.
- Implement secure evidence sharing by generating time-limited signed URLs referencing IPFS CIDs.

### Phase 5 – Overkill Examples

- Multi-cloud Fabric deployment with RAFT ordering service, certificate authorities per org, and automated TLS rotation.
- Integrating zero-knowledge proofs or public blockchain anchoring on every transaction.
- Native rewriting of mobile clients per platform with bespoke design systems.

---

## Run & Integrate Checklist

1. **Start infrastructure services:**

   ```bash
   cd infrastructure
   ipfs daemon &
   ./fabric-network/start.sh  # wrapper around ./network.sh up createChannel -c watheq
   ```

2. **Start the backend API:**

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

3. **Launch the web console:**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Run the mobile app (Flutter example):**

   ```bash
   cd mobile
   flutter pub get
   flutter run
   ```

5. **Branch discipline:** each member opens `feature/<task>` branches, raises a pull request with notes + test evidence, then merges after review.
6. **Weekly integration:** dedicate 60 minutes to review open PRs, ledger/IPFS health, and merge into `main`.

---

## Final Checklists

**Essential Completion:**

- [ ] OCR service returns readable text.
- [ ] Forgery check outputs similarity scores (and optional visualization).
- [ ] Face verification works on sample data.
- [ ] Web console and mobile app submit documents, surface scores, and show ledger receipts.
- [ ] IPFS stores original files and CIDs are retrievable from both clients.
- [ ] SQLite + Fabric ledger persist hashes, transaction IDs, and replay queue states.
- [ ] Runbook, test plan, demo video, and bilingual user guides are complete.

**Bonus Achievements:**

- [ ] Advanced analytics dashboards and automated PDF/CSV exports.
- [ ] Automated Fabric chaincode CI/CD with health probes.
- [ ] Push notifications or SMS flow for ledger status updates.

**Overkill (skip unless turning this into long-term research):**

- [ ] Multi-cloud Fabric deployment with RAFT ordering and hardware-backed keys.
- [ ] Cross-chain notarisation or zero-knowledge proof pipelines.
- [ ] Full event-driven microservices with streaming analytics clusters.

Following this roadmap, even a beginner team can deliver a credible academic showcase for "Watheq" while reserving stretch goals for bonus credit and ignoring overly ambitious tasks that distract from the core demonstration.
