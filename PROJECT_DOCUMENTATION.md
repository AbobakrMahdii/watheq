# Watheq — Full Project Documentation

> **وثّق (Watheq)** — An AI-powered document authenticity verification platform combining deep learning, biometric matching, OCR, blockchain immutability, and a multi-client architecture (mobile + admin dashboard).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Directory Structure](#4-directory-structure)
5. [System Startup Flow (`start_all.bat`)](#5-system-startup-flow-start_allbat)
6. [AI Training Pipeline](#6-ai-training-pipeline)
7. [Document Verification Pipeline](#7-document-verification-pipeline)
8. [Backend API (FastAPI)](#8-backend-api-fastapi)
9. [Mobile Application (Flutter)](#9-mobile-application-flutter)
10. [Admin Dashboard (Next.js)](#10-admin-dashboard-nextjs)
11. [Blockchain & IPFS Layer](#11-blockchain--ipfs-layer)
12. [Biometric Face Verification](#12-biometric-face-verification)
13. [OCR Service](#13-ocr-service)
14. [Database Schema](#14-database-schema)
15. [Security & Authentication](#15-security--authentication)
16. [Audit Logging System](#16-audit-logging-system)
17. [Notification System](#17-notification-system)
18. [Data Flow Diagrams](#18-data-flow-diagrams)
19. [Configuration Files Reference](#19-configuration-files-reference)
20. [Error Handling & Resilience](#20-error-handling--resilience)

---

## 1. Project Overview

**Watheq** is a comprehensive document verification system designed for government-grade identity document authentication. It solves the problem of forged documents by combining multiple verification layers:

| Layer                       | Purpose                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| **AI Visual Verification**  | Binary classifiers (EfficientNet-B0) trained to detect forged vs genuine document elements |
| **Font Analysis**           | Statistical font profile matching to detect text manipulation                              |
| **Biometric Matching**      | Face comparison between the document photo and a live selfie                               |
| **OCR + Data Verification** | Text extraction and cross-referencing against citizen records database                     |
| **Blockchain Recording**    | Immutable verification records on MultiChain + file storage on IPFS                        |
| **Audit Trail**             | Complete audit logging of every system operation                                           |

### What Problems Does Watheq Solve?

1. **Document Forgery Detection** — Uses trained deep learning models to identify tampered logos, seals, barcodes, and text on identity documents
2. **Identity Fraud Prevention** — Matches the face on the document against a live selfie with liveness detection
3. **Data Integrity** — Cross-references OCR-extracted data (name, national ID, dates) against a citizen records database
4. **Immutable Records** — Stores verification results on a private blockchain (MultiChain) and document files on IPFS, creating a tamper-proof audit trail
5. **Administrative Oversight** — Provides a real-time admin dashboard for monitoring verifications, managing users, and viewing analytics

---

## 2. Architecture

### Architectural Pattern: **Modular Layered Monolith**

Watheq follows a **4-tier Modular Layered Monolith** architecture. All backend logic runs within a single FastAPI process, but is organized into clearly separated modules that communicate through well-defined internal interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌─────────────────┐  ┌────────────────────────────────────┐   │
│  │  Flutter Mobile  │  │  Next.js Admin Dashboard           │   │
│  │  (Android/iOS)   │  │  (Web — port 3000)                │   │
│  └────────┬────────┘  └────────────────┬───────────────────┘   │
│           │           HTTP/REST         │                       │
├───────────┴─────────────────────────────┴───────────────────────┤
│                    APPLICATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Backend (port 8012)                             │   │
│  │  ├── Auth Router (/api/v1/auth/*)                        │   │
│  │  ├── Verification Router (/api/v1/verifications/*)       │   │
│  │  ├── Admin Router (/api/v1/admin/*)                      │   │
│  │  ├── Blockchain Router (/api/v1/blockchain/*)            │   │
│  │  ├── Biometric Router (/api/v1/biometric/*)              │   │
│  │  ├── OCR Router (/api/v1/ocr/*)                          │   │
│  │  ├── Document Router (/api/v1/document/*)                │   │
│  │  ├── File Upload Router (/api/v1/files/*)                │   │
│  │  ├── IPFS Router (/api/v1/ipfs/*)                        │   │
│  │  └── Notification Router (/api/v1/notifications/*)       │   │
│  │                                                          │   │
│  │  Services:                                               │   │
│  │  ├── Verification Orchestrator (8-stage pipeline)        │   │
│  │  ├── Verification Steps Service (step implementations)   │   │
│  │  ├── Hash Service (SHA-256)                              │   │
│  │  ├── MultiChain Service (blockchain RPC)                 │   │
│  │  ├── Notification Service (SSE + DB)                     │   │
│  │  ├── Audit Log Service (automatic request logging)       │   │
│  │  ├── File Upload Service (validation + IPFS)             │   │
│  │  └── Document Type Service (CRUD)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                   │
│  ┌─────────────┐  ┌───────────────────┐  ┌─────────────────┐   │
│  │   MySQL      │  │  AI Models        │  │  File Storage   │   │
│  │  (watheq_db) │  │  (PyTorch .pt)    │  │  (storage/)     │   │
│  │  10 tables   │  │  Font Profiles    │  │  verification   │   │
│  │              │  │  (.json)          │  │  artifacts      │   │
│  └─────────────┘  └───────────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    TRUST & STORAGE LAYER                        │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │  MultiChain           │  │  IPFS (Kubo)                  │   │
│  │  (Docker — port 4402) │  │  (Docker — port 15001/18080)  │   │
│  │  Chain: watheqchain   │  │  Decentralized file storage   │   │
│  │  Stream: documents    │  │  Content-addressed (CID)      │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Why "Modular Layered Monolith"?

- **Monolith**: Single deployable unit (one FastAPI process serves everything)
- **Layered**: Clear separation between Presentation → Application → Data → Trust/Storage
- **Modular**: Each domain (AI, Biometric, OCR, Blockchain, Auth) is an independent module with its own directory, models, and interfaces

---

## 3. Technology Stack

### Backend

| Technology           | Version | Purpose                       |
| -------------------- | ------- | ----------------------------- |
| **Python**           | 3.13    | Primary backend language      |
| **FastAPI**          | 0.104.1 | Async REST API framework      |
| **Uvicorn**          | 0.24.0  | ASGI server                   |
| **MySQL**            | 8.x     | Relational database           |
| **aiomysql**         | —       | Async MySQL driver            |
| **databases**        | —       | Async database toolkit        |
| **python-jose**      | 3.3.0   | JWT token creation/validation |
| **passlib + bcrypt** | —       | Password hashing              |
| **PyYAML**           | 6.x     | YAML config parsing           |

### AI / Machine Learning

| Technology      | Version      | Purpose                                             |
| --------------- | ------------ | --------------------------------------------------- |
| **PyTorch**     | 2.6.0+cu124  | Deep learning framework (CUDA 12.4)                 |
| **torchvision** | 0.21.0+cu124 | EfficientNet-B0 pretrained backbone                 |
| **OpenCV**      | 4.10.0       | Image processing, face detection, contour detection |
| **NumPy**       | 2.1.3        | Numerical operations                                |

### Frontend — Mobile

| Technology                 | Version | Purpose                                       |
| -------------------------- | ------- | --------------------------------------------- |
| **Flutter**                | 3.x     | Cross-platform mobile framework (Android/iOS) |
| **Dart**                   | 3.x     | Programming language                          |
| **Dio**                    | —       | HTTP client                                   |
| **flutter_secure_storage** | —       | Secure token storage                          |
| **camera**                 | —       | Document/selfie capture                       |
| **local_auth**             | —       | Biometric auth (fingerprint/face)             |

### Frontend — Admin Dashboard

| Technology       | Version | Purpose                              |
| ---------------- | ------- | ------------------------------------ |
| **Next.js**      | 16      | React framework with App Router      |
| **React**        | 19      | UI library                           |
| **TypeScript**   | —       | Type-safe JavaScript                 |
| **Tailwind CSS** | 4.x     | Utility-first CSS                    |
| **shadcn/ui**    | —       | Component library (Radix primitives) |
| **Recharts**     | —       | Charts and analytics visualizations  |
| **Sonner**       | —       | Toast notifications                  |

### Infrastructure

| Technology         | Version | Purpose                                 |
| ------------------ | ------- | --------------------------------------- |
| **Docker**         | —       | Container runtime for IPFS + MultiChain |
| **Docker Compose** | —       | Multi-container orchestration           |
| **MultiChain**     | 2.x     | Private blockchain (JSON-RPC)           |
| **IPFS (Kubo)**    | latest  | Decentralized file storage              |

### Development Tools

| Tool                      | Purpose                                 |
| ------------------------- | --------------------------------------- |
| **Git**                   | Version control                         |
| **Windows Batch Scripts** | Service orchestration (`scripts/*.bat`) |
| **nvidia-smi**            | GPU detection for CUDA auto-setup       |

---

## 4. Directory Structure

```
watheq/
├── ai/                              # AI/ML Module
│   ├── train_ai.py                  # Training orchestrator (CLI)
│   ├── verify_document.py           # Verification engine (CLI)
│   ├── generate_synthetic_data.py   # Data augmentation + forgery generation
│   ├── models/
│   │   ├── element_classifier.py    # EfficientNet-B0 binary classifier
│   │   ├── font_analyzer.py         # Font profile learning & verification
│   │   ├── siamese_verifier.py      # [DEPRECATED] Siamese network (v2 compat)
│   │   ├── yolo_detector.py         # [DEPRECATED] YOLO detector (v2 compat)
│   │   ├── weights/                 # Trained .pt weight files
│   │   ├── fonts/                   # Learned font profiles (.json)
│   │   └── embeddings/              # [DEPRECATED] Siamese embeddings
│   └── data/
│       ├── refrences/{doc_type}/    # Reference images + layout_config.yaml
│       └── training/{doc_type}/     # Generated training data + config.json
│
├── api/                             # FastAPI Backend
│   ├── app.py                       # Application factory + startup logic
│   ├── main.py                      # Entry point (uvicorn launcher)
│   ├── config.py                    # JWT / env configuration
│   ├── database.py                  # MySQL collections (async ORM-like)
│   ├── models.py                    # Pydantic models
│   ├── security.py                  # JWT auth + RBAC middleware
│   ├── seed.py                      # Database seeder
│   ├── seed_citizens.py             # Citizens records seeder
│   ├── routers/                     # 15 API route modules
│   └── services/                    # Business logic services
│
├── app/                             # Flutter Mobile Application
│   ├── lib/
│   │   ├── main.dart                # App entry point
│   │   ├── core/                    # Config, networking, storage
│   │   ├── features/                # Auth, Verification, Biometric
│   │   ├── screens/                 # UI screens (login, camera, dashboard)
│   │   └── ui/                      # Theme, widgets
│   └── pubspec.yaml                 # Dart dependencies
│
├── dashboard/                       # Next.js Admin Dashboard
│   ├── src/
│   │   ├── app/                     # App Router pages
│   │   │   ├── auth/login/          # Admin login page
│   │   │   ├── dashboard/           # Main dashboard + sub-pages
│   │   │   │   ├── page.tsx         # Analytics overview
│   │   │   │   ├── users/           # User management
│   │   │   │   ├── verifications/   # Verification list + detail
│   │   │   │   ├── blockchain/      # Blockchain operations
│   │   │   │   ├── document-types/  # Document type CRUD
│   │   │   │   ├── audit-logs/      # Audit log viewer + export
│   │   │   │   ├── admins/          # Admin management
│   │   │   │   └── reports/         # Reports page
│   │   │   └── api/                 # Next.js API routes (BFF proxy)
│   │   ├── components/              # Shared UI components
│   │   ├── lib/                     # Utilities (backend proxy, theme)
│   │   └── config/                  # App configuration
│   └── package.json                 # Node.js dependencies
│
├── Biometric/                       # Face verification service
│   └── face_service.py              # DeepFace-based face matching
│
├── ocr/                             # OCR service
│   └── vision_service_ocr.py        # Google Vision / Tesseract OCR
│
├── ledger/                          # IPFS service client
│   └── ipfs_service.py              # HTTP client for Kubo API
│
├── infrastructure/                  # Docker containers
│   ├── docker-compose.ipfs.yml      # IPFS Kubo container
│   ├── docker-compose.multichain.yml# MultiChain container
│   ├── Dockerfile.multichain        # MultiChain image build
│   ├── entrypoint-multichain.sh     # Chain creation + stream setup
│   └── ipfs_service.py              # Infrastructure IPFS service
│
├── core/                            # Shared configuration
│   └── config.py                    # Central config (blockchain RPC, IPFS)
│
├── scripts/                         # Automation scripts
│   ├── start_all.bat                # Start all services (main entry)
│   ├── start_backend.bat            # Start FastAPI server
│   ├── start_dashboard.bat          # Start Next.js dev server
│   ├── start_ipfs.bat               # Start IPFS Docker container
│   ├── start_multichain.bat         # Start MultiChain Docker container
│   ├── setup_python.bat             # Auto-detect GPU + install deps
│   ├── train_ai.bat                 # Run AI training pipeline
│   └── check_backend.bat            # Health-check the backend
│
├── storage/                         # Runtime storage
│   └── verifications/{id}/          # Per-verification artifacts
│       ├── document_front.jpg       # Original upload
│       ├── person_image.jpg         # Selfie upload
│       ├── document_cropped.jpg     # Rectified document
│       ├── document_face.jpg        # Extracted face from document
│       ├── debug/                   # Debug images
│       └── layout/report.json       # Layout gating report
│
├── docs/                            # Documentation
├── utils/                           # Utility scripts
├── requirements.txt                 # Python dependencies (base)
└── requirements.unified.txt         # Full unified requirements
```

---

## 5. System Startup Flow (`start_all.bat`)

When you run `scripts\start_all.bat`, the following happens in order:

### Step 0/5 — AI Model Check

```
[0/5] Checking AI models (v3 — ElementClassifier + FontAnalyzer)...
```

1. **PyTorch Check**: `python -c "import torch"` — If PyTorch is not installed, AI training is skipped with a warning directing the user to run `setup_python.bat`
2. **Training Script**: `python ai\train_ai.py --all` — This script:
   - Discovers all document types in `ai/data/refrences/` that have a `layout_config.yaml`
   - For each document type, checks if all `.pt` weight files already exist
   - If all weights exist → **skips** (instant, <1 second)
   - If any weights are missing → generates augmented data + trains classifiers
   - Also learns font profiles for text regions
3. **Result**: AI models are ready for the verification pipeline

### Step 1/5 — IPFS

```
[1/5] Starting IPFS...
```

1. Opens a **new terminal window** (`start "Watheq - IPFS"`)
2. Removes any stale `ipfs-node` container
3. Runs `docker-compose -p watheq-ipfs -f infrastructure/docker-compose.ipfs.yml up -d`
4. Waits up to 30 seconds for IPFS API on `http://127.0.0.1:15001` to respond
5. **Ports**: 14001 (Swarm), 15001 (API), 18080 (Gateway)

### Step 2/5 — MultiChain

```
[2/5] Starting MultiChain Blockchain...
```

1. Opens a **new terminal window**
2. Builds and starts the MultiChain Docker container
3. On first run, the entrypoint script (`entrypoint-multichain.sh`):
   - Creates blockchain `watheqchain`
   - Configures RPC credentials (`watheqrpc:watheqrpcpass`)
   - Creates a `documents` stream and subscribes to it
4. Waits up to 40 seconds for RPC on `http://127.0.0.1:4402` to respond
5. **Port**: 4402 (JSON-RPC), 4403 (Peer-to-peer)

### Step 3/5 — Backend API

```
[3/5] Starting Backend API...
```

1. Opens a **new persistent terminal window** (`cmd /k`)
2. Creates `.venv` if not present (using Python 3.13)
3. Installs dependencies from `requirements.unified.txt` (one-time)
4. Runs `python -u -m api.main` → Starts Uvicorn on `0.0.0.0:8012`
5. **On startup** (`app.py` → `startup_event()`):
   - Auto-creates `watheq_db` database if it doesn't exist
   - Connects to MySQL
   - Creates all 10 tables with `CREATE TABLE IF NOT EXISTS`
   - Runs best-effort `ALTER TABLE` migrations for schema evolution
   - Seeds default super admin (`admin@admin.admin` / `pass1234`)
6. **Port**: 8012 (API), docs at `http://localhost:8012/api/v1/docs`

### Step 4/5 — Dashboard

```
[4/5] Starting Dashboard...
```

1. Opens a **new persistent terminal window**
2. Installs `node_modules` if not present (`npm install`)
3. Creates `.env.local` from example if not present
4. Runs `npm run dev` → Starts Next.js on `http://localhost:3000`
5. **Port**: 3000

### Final Output

```
Services:
  - IPFS:        http://localhost:15001 (API), http://localhost:18080 (Gateway)
  - MultiChain:  http://localhost:4402 (JSON-RPC) - watheqchain
  - Backend API: http://localhost:8012
  - API Docs:    http://localhost:8012/api/v1/docs
  - Dashboard:   http://localhost:3000
```

---

## 6. AI Training Pipeline

### Overview

The AI training pipeline creates per-element binary classifiers that learn to distinguish genuine document elements from forgeries **without requiring reference images at runtime**.

### Pipeline Steps (in `ai/train_ai.py`)

#### Step 1: Discovery

```python
discover_doc_types()  # Scans ai/data/refrences/ for folders with layout_config.yaml
```

Currently supported: `identity` (Yemeni National ID Card)

#### Step 2: Layout Config Loading

Each document type has a `layout_config.yaml` that defines:

```yaml
elements:
  logo: # Reference image stem name
    class_name: logo_main # Internal classifier name
    ref_file: logo.png # Path to reference image
    roi: # Normalized position on document
      x: 0.02
      y: 0.03
      w: 0.18
      h: 0.35
    tolerance: 0.08
    weight: 1.5 # Importance in final score
    critical: true # Failure = automatic document failure
    type: visual

text_regions:
  text_name:
    class_name: text_name
    roi: { x: 0.35, y: 0.25, w: 0.35, h: 0.08 }
    type: text

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  val_split: 0.2
  early_stopping_patience: 5

thresholds:
  pass_score: 0.95 # Minimum to pass verification
  suspicious_score: 0.70 # Below this = definite fail
```

#### Step 3: Synthetic Data Generation

For each visual element (e.g., `logo.png`):

```
generate_synthetic_data.py
├── Input: clean reference image (e.g., logo.png)
├── Genuine augmentations (400 images):
│   ├── Random rotation (±7°)
│   ├── Random scale (0.92–1.08)
│   ├── Random translation (±3%)
│   ├── Gaussian blur (kernel 3 or 5)
│   ├── Gaussian noise (σ=2–7)
│   ├── Brightness/contrast jitter
│   ├── HSV color jitter
│   └── JPEG compression artifacts (quality 60–85)
│
└── Forged samples (400 images, 4 types × 100 each):
    ├── Freehand (ff_): Adaptive threshold → ink trace
    ├── Tracing (ft_): Canny edges → dilated line drawing
    ├── Digital (fd_): Resize + sharpening + heavy JPEG compression
    └── Color Forge (fc_): Channel swap / heavy tint / desaturation
```

Output: `ai/data/training/identity/{element}/out_genuine/*.png` and `out_forged/*.png`

#### Step 4: Classifier Training

For each visual element, an `ElementClassifier` is trained:

**Architecture**: EfficientNet-B0 (pretrained ImageNet) → AdaptiveAvgPool → Flatten → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→1)

**Training Strategy**:

- **Epochs 0–1**: Backbone frozen, only head trains (transfer learning)
- **Epoch 2+**: Full fine-tuning with 10× lower learning rate
- **Loss**: `BCEWithLogitsLoss` (AMP-safe, numerically stable)
- **Optimizer**: AdamW with ReduceLROnPlateau scheduler
- **Early Stopping**: Patience of 5 epochs on validation loss
- **Mixed Precision**: Automatic Mixed Precision (AMP) with `GradScaler` on GPU
- **Data Loading**: 4 workers, persistent, pinned memory, batch size 64 on GPU

**GPU Optimization**:

- Auto-detects CUDA GPU (RTX 3050 Ti, 4GB VRAM)
- `cudnn.benchmark = True` for fixed input size optimization
- `non_blocking=True` for async CPU→GPU transfers
- `set_to_none=True` for memory-efficient gradient zeroing
- Module-level `_ElementDataset` for Windows multiprocessing compatibility

**Output**: `ai/models/weights/{doc_type}_{class_name}.pt`

#### Step 5: Font Profile Learning

For text regions (e.g., `text_name`, `text_national_id`):

```python
FontAnalyzer.learn_font_profile(text_crop, class_name, doc_type)
```

Learns statistical properties:

- **Ink density** — Ratio of dark pixels to total
- **Stroke width** (mean + std) — From distance transform of binarized text
- **Character height** (mean + std) — Connected component analysis
- **Sharpness** — Laplacian variance
- **Histogram profile** — 16-bin intensity distribution
- **Text uniformity** — Standard deviation of row-wise ink densities

**Output**: `ai/models/fonts/{doc_type}_{class_name}.json`

#### Step 6: Config Save

```json
{
  "doc_type": "identity",
  "version": "3.0",
  "model": "ElementClassifier (EfficientNet-B0 binary)",
  "trained_at": "2026-02-08T...",
  "layout": { ... learned positions ... },
  "thresholds": { "pass_score": 0.95, "suspicious_score": 0.70 },
  "classifier_results": { ... per-element training stats ... },
  "font_results": { ... per-text-region profiles ... }
}
```

---

## 7. Document Verification Pipeline

When a user submits a document for verification, an **8-stage pipeline** runs asynchronously in the background.

### The Verification Orchestrator (`api/services/verification_orchestrator.py`)

```
POST /api/v1/verifications/start
├── Upload: document_image_front, person_image, document_type_id
├── Creates verification record (status: PENDING)
├── Saves uploads to storage/verifications/{id}/
└── Launches background pipeline ──────────────────────────┐
                                                           │
    ┌──────────────────────────────────────────────────────┘
    │
    ▼
Stage 1: IMAGE_QUALITY_CHECK
    ├── Brightness check (40–220 range)
    ├── Blur detection (Laplacian variance ≥ 70)
    └── Result: { brightness, blur_score, ok/fail }
    │
    ▼
Stage 2: DOCUMENT_CROPPING
    ├── Contour detection → find document quadrilateral
    ├── Perspective warp (rectification)
    ├── Fallback: min-area rect → fallback: center crop
    └── Output: storage/{id}/document_cropped.jpg
    │
    ▼
Stage 3: LAYOUT_GATING
    ├── Aspect ratio check (1.2–2.1 for ID cards)
    ├── Edge density check (>2% non-zero edges)
    ├── Face presence check (Haar cascade)
    ├── Minimum resolution (200×120)
    └── Output: storage/{id}/layout/report.json
    │
    ▼
Stage 4: DOCUMENT_FACE_EXTRACTION
    ├── Try layout ROI from template first
    ├── Fallback: OpenCV Haar cascade face detection
    ├── Largest face + 15% margin padding
    └── Output: storage/{id}/document_face.jpg
    │
    ▼
Stage 5: FACE_MATCHING
    ├── DeepFace verification (document face ↔ selfie)
    ├── Uses ArcFace / VGG-Face / Facenet models
    ├── cosine distance threshold for match/no-match
    └── Result: { match: bool, confidence: float }
    │
    ▼
Stage 6: ML_VERIFICATION (AI)
    ├── Runs ai/verify_document.py as subprocess
    ├── For each visual element:
    │   ├── Crop from rectified document using ROI
    │   ├── Run trained ElementClassifier → probability
    │   └── Score = 70% classifier + 30% position
    ├── For each text element:
    │   ├── Crop text region
    │   ├── Run FontAnalyzer.verify_font() → font score
    │   └── Score = 60% font + 40% position
    ├── Weighted overall confidence across all elements
    └── Decision: PASSED (≥95%) / SUSPICIOUS (≥70%) / FAILED
    │
    ▼
Stage 7: OCR_VERIFY
    ├── Google Vision / Tesseract OCR on document
    ├── Extract: national_id, full_name_ar, dates
    └── Result: { text, fields }
    │
    ▼
Stage 8: DATA_VERIFICATION
    ├── Query citizen_records by extracted national_id
    ├── Compare: name, DOB, issue/expiry dates
    └── Result: { citizen_found, match_count, match_details }
    │
    ▼
Stage 9: BLOCKCHAIN_RECORDING
    ├── SHA-256 hash of document file
    ├── Pin file to IPFS → get CID
    ├── Publish metadata to MultiChain "documents" stream
    │   { doc_id, cid, sha256, owner, timestamp }
    └── Result: { doc_id, cid, sha256, ledger_recorded: true }
    │
    ▼
FINAL: Pipeline Complete
    ├── Status: SUCCESS / FAILED
    ├── If FAILED → create admin notification
    └── All step results saved to verification_steps table
```

### What the User Sees (Mobile App)

1. **Document Capture Screen** → Camera preview with rectangle overlay → Auto-capture
2. **Selfie/Liveness Screen** → Front camera → Checks for blinking/head movement
3. **Loading/Progress** → Real-time polling of verification status
4. **Result Screen** → Shows PASSED ✓ / FAILED ✗ with per-element breakdown and confidence percentages

### What the Admin Sees (Dashboard)

1. **Verifications List** → Filterable table with status, user, document type, date
2. **Verification Detail** → All 8 pipeline stages with individual step results, timings, and error messages
3. **Admin Notes** → Can add notes to any verification for auditing
4. **Notifications** → Real-time SSE alerts for failed verifications

---

## 8. Backend API (FastAPI)

### Entry Point

```python
# api/main.py
uvicorn.run(app, host="0.0.0.0", port=8012)
```

### Application Factory (`api/app.py`)

The `app.py` file creates the FastAPI application with:

1. **CORS Middleware** — Allows `localhost:3000` in development
2. **Audit Middleware** — Logs every HTTP request (success/failure/exception)
3. **15 Routers** — Organized by domain
4. **OpenAPI Security** — Bearer JWT auth in Swagger UI
5. **Startup Event** — Auto-creates database, tables, migrations, default admin

### Router Map

| Router                       | Prefix                      | Auth   | Purpose                                 |
| ---------------------------- | --------------------------- | ------ | --------------------------------------- |
| `auth_router`                | `/api/v1/auth`              | None   | Register, Login, Logout, Me             |
| `admin_router`               | `/api/v1/admin`             | Admin  | User CRUD, Promote, Suspend, Edit       |
| `verification_router`        | `/api/v1/verifications`     | User   | Start verification, List my, Get status |
| `admin_verification_router`  | `/api/admin/verifications`  | Admin  | List all, Stats, Notes                  |
| `document_router`            | `/api/v1/document`          | User   | Direct document verify                  |
| `blockchain_router`          | `/api/v1/blockchain`        | Varies | Chain info, Streams, Publish, Verify    |
| `biometric_router`           | `/api/v1/biometric`         | User   | Face verify + liveness                  |
| `face_router`                | `/api/v1/face`              | User   | Face comparison endpoint                |
| `ocr_router`                 | `/api/v1/ocr`               | User   | OCR extraction                          |
| `ipfs_router`                | `/api/v1/ipfs`              | User   | Pin/get files on IPFS                   |
| `file_upload_router`         | `/api/v1/files`             | User   | Generic file upload                     |
| `document_type_router`       | `/api/document-types`       | None   | List active document types              |
| `admin_document_type_router` | `/api/admin/document-types` | Admin  | CRUD document types                     |
| `admin_audit_router`         | `/api/admin/audit-logs`     | Admin  | Audit log list + export                 |
| `notification_router`        | `/api/v1/notifications`     | Admin  | SSE stream + CRUD notifications         |

### Key API Endpoints

#### Authentication

```
POST /api/v1/auth/register     → { name, username, email, password }
POST /api/v1/auth/login        → Form: { username, password } → { access_token, role }
GET  /api/v1/auth/me           → Current user profile
POST /api/v1/auth/logout       → Invalidate session (client-side)
```

#### Verification

```
POST /api/v1/verifications/start
  → Form-data: document_image_front, person_image, document_type_id
  → Returns: VerificationPublic (id, status: PENDING)
  → Pipeline runs in background

GET  /api/v1/verifications/{id}       → Verification status + result_data
GET  /api/v1/verifications/{id}/steps → All pipeline steps with details
GET  /api/v1/verifications/my         → Paginated list + status counts
```

#### Admin

```
GET    /api/v1/admin/users                → List all users
POST   /api/v1/admin/users/create         → Create user
PUT    /api/v1/admin/users/{id}           → Edit user (name, email, password)
PUT    /api/v1/admin/users/{id}/make-admin → Promote to admin
PUT    /api/v1/admin/users/{id}/suspend    → Soft-delete (suspend)
PUT    /api/v1/admin/users/{id}/activate   → Reactivate suspended user
GET    /api/admin/verifications            → All verifications (filtered)
GET    /api/admin/analytics                → Dashboard analytics (counts, charts)
GET    /api/admin/audit-logs               → Paginated audit logs
GET    /api/admin/audit-logs/export        → Export as PDF or XLSX
```

---

## 9. Mobile Application (Flutter)

### App Structure

```dart
main() → MyApp → MaterialApp
  routes:
    '/'           → SplashScreen (checks saved token)
    '/login'      → LoginScreen
    '/home'       → HomeScreen
    '/dashboard'  → DashboardScreen (bottom nav)
```

### Screen Flow

1. **SplashScreen** — Checks `SecureStorage` for saved JWT token → validates with `/api/v1/auth/me` → navigates to `/dashboard` or `/login`

2. **LoginScreen** — Email/password form → `POST /api/v1/auth/login` → saves token to `flutter_secure_storage` → navigates to dashboard

3. **DashboardScreen** — Bottom navigation with 3 tabs:
   - **HomeScreen** — Welcome + quick actions
   - **VerificationHistoryScreen** — List of past verifications with status badges
   - **ProfileScreen** — User profile info

4. **Verification Flow** (triggered by FAB button):
   - **DocumentCaptureScreen** → Opens camera → Auto-captures document when steady
   - **SelfieLivenessScreen** → Front camera → Captures selfie with liveness cues
   - Calls `VerificationOrchestratorService.start()` → `POST /api/v1/verifications/start`
   - **VerificationResultScreen** → Polls status → Shows pass/fail with element breakdown

### Key Services

| Service                           | Purpose                                                    |
| --------------------------------- | ---------------------------------------------------------- |
| `AuthService`                     | Login, logout, token management via `SecureStorageService` |
| `VerificationOrchestratorService` | Start verification, get status/steps, list history         |
| `DocumentVerifyService`           | Direct document verification (standalone)                  |
| `FaceVerifyService`               | Face comparison (document photo vs selfie)                 |
| `OcrService`                      | OCR text extraction                                        |
| `IpfsService`                     | IPFS file management                                       |
| `NotificationService`             | Local push notifications                                   |
| `VerificationTracker`             | Background polling to resume interrupted verifications     |

### Network Architecture

- All API calls go through `ApiClient` (Dio-based)
- `AuthInterceptor` automatically handles 401 → navigates to login
- Base URL: `http://192.168.8.36:8012` (configurable in `AppConfig`)

---

## 10. Admin Dashboard (Next.js)

### Architecture

The dashboard uses Next.js **App Router** with a Backend-for-Frontend (BFF) pattern:

```
Browser → Next.js API Routes (/api/*) → FastAPI Backend (:8012)
```

This means the browser never directly calls the Python backend. Next.js API routes act as a proxy, handling:

- Cookie-based session management (httpOnly cookies)
- Token forwarding via Authorization header
- Error normalization

### Page Map

| Page                | Route                           | Features                                                            |
| ------------------- | ------------------------------- | ------------------------------------------------------------------- |
| Login               | `/auth/login`                   | Email/username + password                                           |
| Overview            | `/dashboard`                    | 6 stat cards + pie chart + bar chart + line chart + failure reasons |
| Users               | `/dashboard/users`              | User table, create, edit (dialog), suspend, activate, promote       |
| Verifications       | `/dashboard/verifications`      | Filterable table, status badges, pagination                         |
| Verification Detail | `/dashboard/verifications/[id]` | 8-stage pipeline view, step results, admin notes                    |
| Document Types      | `/dashboard/document-types`     | CRUD with name, folder, active toggle                               |
| Blockchain          | `/dashboard/blockchain`         | Chain info, streams, documents, IPFS, peers, permissions            |
| Audit Logs          | `/dashboard/audit-logs`         | Filterable log table, export PDF/XLSX                               |
| Admins              | `/dashboard/admins`             | Admin-only user management                                          |
| Reports             | `/dashboard/reports`            | Report generation                                                   |

### Key Components

| Component              | Purpose                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `AuthGuard`            | Wraps all dashboard pages; redirects to login if not authenticated |
| `DashboardSidebar`     | Navigation sidebar with links and active state                     |
| `NotificationProvider` | SSE connection for real-time admin alerts                          |
| `NotificationBell`     | Bell icon with unread count badge                                  |
| `ProfileNav`           | Current user info + logout button                                  |

### Dashboard-Backend Proxy

The file `dashboard/src/lib/backend.ts` provides:

```typescript
backendGet(path); // GET  → http://localhost:8012/{path} + auth header
backendPost(path); // POST → http://localhost:8012/{path} + auth header
backendPut(path); // PUT  → http://localhost:8012/{path} + auth header
backendDelete(path); // DELETE → http://localhost:8012/{path} + auth header
```

Each Next.js API route (`/api/admin/users/route.ts`) calls these functions, forwarding the `watheq_token` cookie as a Bearer token.

---

## 11. Blockchain & IPFS Layer

### MultiChain

**Purpose**: Immutable ledger for verification records.

**Setup**:

- Docker container from custom `Dockerfile.multichain`
- Chain: `watheqchain`
- RPC: `http://127.0.0.1:4402` (user: `watheqrpc`, pass: `watheqrpcpass`)
- Auto-created stream: `documents`

**How it's used**:

1. After successful verification, the orchestrator calls `blockchain_verify()`
2. Document file is SHA-256 hashed
3. File is pinned to IPFS → gets a CID
4. Metadata (doc_id, CID, SHA-256, owner, timestamp) is published to the `documents` stream
5. This creates an immutable record that can be independently verified

**MultiChain Service** (`api/services/multichain_service.py`):

```python
publish_to_stream(key, hex_data)  # Publishes to "documents" stream
json_to_hex(json_string)          # Converts JSON to hex for MultiChain
```

### IPFS

**Purpose**: Decentralized, content-addressed file storage.

**Setup**:

- Docker container: `ipfs/kubo:latest`
- API: `http://127.0.0.1:15001`
- Gateway: `http://127.0.0.1:18080`

**Operations**:

```python
ipfs_service = IPFSService(base_url="http://127.0.0.1:15001/api/v0")
cid = ipfs_service.pin_file("/path/to/document.jpg")  # Upload + pin
data = ipfs_service.cat(cid)                           # Retrieve by CID
node_id = ipfs_service.id()                            # Node identity
```

---

## 12. Biometric Face Verification

### Service (`Biometric/face_service.py`)

Uses **DeepFace** library for face comparison between the document photo and the live selfie.

**Process**:

1. Document face is extracted in Stage 4 (DOCUMENT_FACE_EXTRACTION)
2. Live selfie is provided by the user via the Flutter app
3. `FaceService.verify_id_vs_live(doc_bytes, selfie_bytes)` compares them
4. Returns: match (bool), confidence score, distance metric

**Methods used**: ArcFace, VGG-Face, or Facenet (depending on availability)

---

## 13. OCR Service

### Service (`ocr/vision_service_ocr.py`)

Provides two functions:

- `ocr_image(image_bytes)` — Extracts text from an image
- `ocr_pdf(pdf_bytes, max_pages)` — Extracts text from PDF pages

**Post-processing** (in `data_verification`):

- Regex extraction for national ID (8-12 digit numbers)
- Arabic name pattern matching
- Date extraction (dd/mm/yyyy or dd-mm-yyyy format)

---

## 14. Database Schema

MySQL database: `watheq_db`

### Tables

| Table                 | Purpose                      | Key Columns                                                                                    |
| --------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `users`               | User accounts                | id, name, username, email, password (bcrypt), role, is_active, deleted_at                      |
| `document_types`      | Document type definitions    | id, name, folder_name, is_active, requires_back_image                                          |
| `verifications`       | Verification records         | id, user_id, document_type_id, status, current_stage, result_data (JSON), start_time, end_time |
| `verification_steps`  | Pipeline step results        | id, verification_id, step_name, stage, status, result_data (JSON)                              |
| `verification_notes`  | Admin notes on verifications | id, verification_id, admin_id, note_text                                                       |
| `document_hashes`     | SHA-256 + IPFS CID records   | id, document_id, hash (unique), ipfs_cid                                                       |
| `audit_logs`          | Complete audit trail         | 21 columns: operation_id, type, status, user info, IP, path, method, file info, extra_data     |
| `biometric_audit_log` | Face match results           | id, user_id, document_id, liveness_result, match_result, confidence_score                      |
| `citizen_records`     | Reference citizen data       | national_id (unique), names, DOB, dates, address                                               |
| `notifications`       | Admin failure alerts         | id, verification_id, message, failure_stage, is_read                                           |

### Roles

| Role          | Permissions                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- |
| `user`        | Register, login, start verifications, view own history                                        |
| `admin`       | All user permissions + manage users, view all verifications, add notes, manage document types |
| `super_admin` | All admin permissions + promote/demote admins, create admins                                  |

---

## 15. Security & Authentication

### JWT Authentication

1. **Login** → `POST /api/v1/auth/login` → Returns `{ access_token, token_type: "bearer", role }`
2. **Token Content**: `{ sub: user_id, email, role }` signed with HS256
3. **Token Lifetime**: 60 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
4. **Secret**: Auto-generated in development, required via `SECRET_KEY` env var in production

### Request Authentication Flow

```
Client Request
  → Authorization: Bearer <jwt_token>
    → security.py::get_current_user()
      → Decode JWT → Verify signature → Check expiry
        → Return payload { sub, email, role }

For admin routes:
  → security.py::get_current_admin()
    → get_current_user() + check role ∈ { "admin", "super_admin" }
```

### Password Security

- Hashing: `passlib[bcrypt]` with bcrypt scheme
- Legacy support: Auto-migrates plaintext passwords to bcrypt on first login
- Minimum validation: Server-side (no empty fields)

### Dashboard Session

- The Next.js dashboard uses **httpOnly cookies** (`watheq_token`)
- On login: `POST /api/auth/login` → Next.js calls FastAPI → Sets cookie
- On each request: Middleware reads cookie → Forwards as Bearer token to FastAPI
- `AuthGuard` component redirects to `/auth/login` if no valid session

---

## 16. Audit Logging System

Every HTTP request to the backend is automatically logged by the audit middleware.

### What Gets Logged

| Field                        | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `operation_id`               | UUID per request                                           |
| `operation_type`             | Inferred from path (Login, FileUpload, Verification, etc.) |
| `status`                     | success / failed                                           |
| `failure_reason`             | Error message if failed                                    |
| `user_id/name/email/role`    | Authenticated user info                                    |
| `ip_address`                 | Client IP                                                  |
| `user_agent`                 | Browser/client string                                      |
| `path` / `method`            | HTTP route and method                                      |
| `file_name/ext/size/cid/url` | For file operations                                        |
| `extra_data`                 | JSON blob for additional context                           |
| `created_at`                 | Timestamp                                                  |

### Export

Admins can export audit logs as:

- **PDF** — `GET /api/admin/audit-logs/export?format=pdf`
- **Excel** — `GET /api/admin/audit-logs/export?format=xlsx`

---

## 17. Notification System

### Architecture

- **Database-backed** notifications stored in `notifications` table
- **Server-Sent Events (SSE)** for real-time push to admin dashboard
- **Triggered** when a verification fails (via `notification_service.py`)

### Flow

```
Verification Pipeline FAILS
  → notification_service.create_notification(verification_id, message, ...)
    → INSERT INTO notifications
    → SSE broadcast to all connected admin clients

Admin Dashboard (NotificationProvider)
  → Connects to GET /api/v1/notifications/stream (SSE)
    → Receives real-time events
    → Updates NotificationBell count badge
    → Shows toast notification
```

### Endpoints

```
GET    /api/v1/notifications/stream      → SSE event stream
GET    /api/v1/notifications             → List (paginated)
GET    /api/v1/notifications/unread-count → Count of unread
PATCH  /api/v1/notifications/{id}/read   → Mark one as read
PATCH  /api/v1/notifications/read-all    → Mark all as read
```

---

## 18. Data Flow Diagrams

### User Verification Flow (Mobile → Backend → AI → Blockchain)

```
[Flutter App]                    [FastAPI Backend]               [External Services]
     │                                  │                              │
     │ 1. Open camera                   │                              │
     │ 2. Capture document front        │                              │
     │ 3. Capture selfie                │                              │
     │                                  │                              │
     │── POST /verifications/start ────→│                              │
     │   (front.jpg + selfie.jpg +      │                              │
     │    document_type_id)             │                              │
     │                                  │── Save files to storage/     │
     │                                  │── INSERT verification (PENDING)
     │                                  │── Launch background task ────┐
     │←── { id, status: PENDING } ──────│                              │
     │                                  │                              │
     │ 4. Poll GET /verifications/{id}  │                              │
     │                                  │    ┌─────────────────────────┘
     │                                  │    │ Background Pipeline:
     │                                  │    │
     │                                  │    ├── Stage 1: Quality → OK
     │                                  │    ├── Stage 2: Crop → Rectify
     │                                  │    ├── Stage 3: Layout → PASS
     │                                  │    ├── Stage 4: Face Extract
     │                                  │    ├── Stage 5: Face Match ──→ [DeepFace]
     │                                  │    ├── Stage 6: AI Verify ──→ [PyTorch Models]
     │                                  │    ├── Stage 7: OCR ────────→ [Vision API]
     │                                  │    ├── Stage 8: Data Verify → [MySQL Citizens]
     │                                  │    └── Stage 9: Blockchain ─→ [IPFS] + [MultiChain]
     │                                  │    │
     │                                  │    └── UPDATE verification → SUCCESS/FAILED
     │                                  │                              │
     │←── { status: SUCCESS, results } ─│                              │
     │                                  │    If FAILED → notify admins │
     │ 5. Show result to user           │                              │
```

### Admin Dashboard Data Flow

```
[Admin Browser]           [Next.js BFF]              [FastAPI Backend]
     │                        │                             │
     │── GET /dashboard ──→   │                             │
     │                        │── GET /api/admin/analytics  │
     │                        │   (Cookie → Bearer token)  │
     │                        │────────────────────────────→│
     │                        │                             │── Query MySQL
     │                        │←── { stats JSON } ─────────│
     │←── Rendered page ──────│                             │
     │                        │                             │
     │   SSE /notifications   │                             │
     │←──────── stream ───────│←── SSE proxy ──────────────│
     │   { new_failed_verif } │                             │
```

---

## 19. Configuration Files Reference

### `ai/data/refrences/{doc_type}/layout_config.yaml`

Defines document layout, element positions, training parameters, and thresholds. This is the **single source of truth** for what a document type looks like.

### `ai/data/training/{doc_type}/config.json`

Auto-generated after training. Records training results, timestamps, and configuration version.

### `api/.env` (optional)

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=watheq_db
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### `dashboard/.env.local`

```env
BACKEND_BASE_URL=http://localhost:8012
```

### `core/config.py`

Central configuration for blockchain and IPFS service URLs.

### `app/lib/core/config/app_config.dart`

```dart
static const String apiBaseUrl = 'http://192.168.8.36:8012';
```

---

## 20. Error Handling & Resilience

### Backend Error Handling

- **HTTP Exceptions** — Custom handlers log to audit before returning error response
- **Validation Errors** — Pydantic validation errors are caught and logged
- **Startup Failures** — Non-fatal: tables/columns that already exist are silently skipped
- **Service Failures** — Each verification stage catches exceptions and records them in the step result

### Verification Pipeline Resilience

- Each stage is wrapped in try/except
- If a stage fails, the failure is recorded but the pipeline can continue or abort
- Verification status is updated to `FAILED` with the error message
- Failed verifications trigger admin notifications

### Flutter Error Handling

- `NetworkExceptions.toUserMessage(e)` converts Dio exceptions to Arabic user messages
- `AuthInterceptor` catches 401 and redirects to login
- Token validation on app startup with fallback to login

### Dashboard Error Handling

- `AuthGuard` validates session on every page load
- Toast notifications (`Sonner`) for user-facing errors
- Graceful fallbacks for missing data (empty states, skeleton loading)

---

_This document was auto-generated from deep analysis of the Watheq project source code._
