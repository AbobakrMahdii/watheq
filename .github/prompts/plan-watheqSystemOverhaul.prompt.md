# Plan: Watheq Full System Overhaul

**TL;DR**: Restructure the entire verification pipeline into 4 clean layers (Biometric → OCR → AI → Blockchain), replace SSIM-based document verification with YOLOv8 + Siamese Network deep learning, remove Hyperledger Fabric (keep MultiChain only), add citizen records DB verification, build admin verification page with full filters + notes, expand analytics with charts and exports, and update the audit report to reflect all requirement changes (deletions, deferrals, and new scope).

---

## Phase 1: Cleanup & Removals

### 1.1 Remove Hyperledger Fabric from entire codebase

- Delete `api/services/fabric_service.py` entirely
- Delete `api/routers/ledger_router.py` entirely
- Remove Fabric import and router include from `api/app.py`
- Remove `fabric_invoke` / `fabric_query` calls from `api/services/verification_steps_service.py` — the `blockchain_verify` function should use MultiChain only
- Delete `ledger/` folder (contains IPFS service that's duplicated — keep only `infrastructure/ipfs_service.py` or consolidate into one)
- Remove any Fabric-related Flutter services (check `LedgerService` in app)
- Remove ledger-related dashboard API proxy routes

### 1.2 Remove redundant BIOMETRIC pipeline stage

- In `api/services/verification_orchestrator.py`: remove the `BIOMETRIC` stage (stage 6) — it duplicates `FACE_MATCHING` (stage 5)
- Update `VerificationStage` enum in `api/models.py`: remove `BIOMETRIC`
- Keep the `biometric_router.py` endpoint for standalone face comparison if needed, but remove it from the pipeline

### 1.3 Remove unused code

- Remove server-side liveness check in `api/services/liveness_service.py` — client-side ML Kit liveness is sufficient per user decision
- Remove `SELFIE_LIVENESS` stage from pipeline (Flutter handles this client-side; a passed liveness flag is still sent but no server-side re-check needed)

---

## Phase 2: Database Schema Changes

### 2.1 Create `citizen_records` table in `api/database.py`

```
citizen_records:
  id (INT AUTO_INCREMENT PRIMARY KEY)
  national_id (VARCHAR(20) UNIQUE) -- الرقم الوطني
  full_name_ar (VARCHAR(255))      -- الاسم الكامل بالعربية
  full_name_en (VARCHAR(255))      -- الاسم بالإنجليزية
  date_of_birth (DATE)             -- تاريخ الميلاد
  address (TEXT)                   -- العنوان
  issue_date (DATE)                -- تاريخ الإصدار
  expiry_date (DATE)               -- تاريخ الانتهاء
  gender (VARCHAR(10))             -- الجنس
  nationality (VARCHAR(100))       -- الجنسية
  document_type (VARCHAR(50))      -- نوع الوثيقة
  created_at (TIMESTAMP DEFAULT NOW())
  updated_at (TIMESTAMP NULL ON UPDATE NOW())
```

- Add CRUD functions: `get_citizen_by_national_id()`, `create_citizen_record()`, `update_citizen_record()`
- Create seed script with ~20 mock citizen records for demo

### 2.2 Create `verification_notes` table

```
verification_notes:
  id (INT AUTO_INCREMENT PRIMARY KEY)
  verification_id (INT, FK → verifications.id)
  admin_id (INT, FK → users.id)
  note_text (TEXT NOT NULL)
  created_at (TIMESTAMP DEFAULT NOW())
```

- Add CRUD functions: `add_verification_note()`, `get_notes_by_verification_id()`

### 2.3 Add expanded filter support to verifications queries

- Modify `get_all_verifications()` in database.py to accept: `status`, `document_type_id`, `user_id`, `date_from`, `date_to`, `search` (search in user name/email), `sort_by`, `sort_order`

---

## Phase 3: AI Module Complete Overhaul

This is the largest and most critical change. Replace SSIM-based comparison with deep learning.

### 3.1 YOLOv8 Object Detection Model

- **Purpose**: Detect and localize all elements within a document image
- **Elements to detect** (annotation classes):
  1. `logo_main` — الشعار الرئيسي
  2. `logo_secondary` — الشعار في مواقع أخرى (e.g., behind national ID number)
  3. `stamp` — الختم
  4. `photo_primary` — صورة الشخص الرئيسية
  5. `photo_ghost` — صورة الشخص الشفافة (watermark/ghost)
  6. `text_name` — منطقة الاسم
  7. `text_national_id` — منطقة الرقم الوطني
  8. `text_dob` — تاريخ الميلاد
  9. `text_issue_date` — تاريخ الإصدار
  10. `text_expiry_date` — تاريخ الانتهاء
  11. `barcode` — الباركود
  12. `background_pattern` — نمط الخلفية
- **Training data**: Use existing reference images + augmented synthetic data (rotation, brightness, blur, noise) to generate training set. Need ~200+ annotated images per document type (use `ai/augment_genuine.py` + manual annotation via LabelImg/Roboflow)
- **Output**: Bounding boxes + class labels + confidence for each detected element
- **File**: Create `ai/models/yolo_detector.py`

### 3.2 Siamese Network Verification Model

- **Purpose**: For each detected element, verify authenticity against trained reference embeddings
- **Architecture**: Twin CNN branches (EfficientNet-B0 backbone), shared weights, producing 128-dim embeddings. Contrastive loss training.
- **Training pairs**:
  - Positive pairs: augmented versions of genuine elements (same element, different augmentations)
  - Negative pairs: genuine vs forged/tampered elements
- **Per-element verification**: Crop each detected element using YOLO bbox → pass through Siamese net → compare embedding distance against threshold → return similarity score
- **File**: Create `ai/models/siamese_verifier.py`

### 3.3 New `ai/verify_document.py` (complete rewrite)

- Input: rectified document image, document type
- Process:
  1. Run YOLOv8 detection → get all element bounding boxes
  2. **Position validation**: Check each detected element's position against expected layout (within tolerance %). Flag missing elements.
  3. **Size validation**: Check element dimensions against expected ranges
  4. **Siamese verification**: For each detected element, crop and verify against reference embeddings
  5. **Color analysis**: For elements with expected colors (logo, stamp), extract dominant colors and compare
  6. **Text analysis**: For text regions, validate font characteristics (size uniformity, alignment)
  7. **Ghost image check**: Verify presence of watermark/ghost photo, check opacity level
  8. **Overall decision**: Weighted aggregation of all element scores
- Output (detailed JSON):

```json
{
  "decision": "PASSED|FAILED",
  "overall_confidence": 0.94,
  "elements": {
    "logo_main": {
      "detected": true,
      "position": { "x": 120, "y": 50, "w": 80, "h": 80 },
      "position_valid": true,
      "size_valid": true,
      "authenticity_score": 0.96,
      "color_match": 0.93,
      "details": "Logo detected at expected position, authentic"
    },
    "photo_ghost": {
      "detected": true,
      "position": { "x": 300, "y": 100, "w": 60, "h": 80 },
      "position_valid": true,
      "opacity_detected": 0.35,
      "details": "Ghost image present with expected transparency"
    }
  },
  "missing_elements": [],
  "anomalies": [],
  "processing_time_ms": 1200
}
```

### 3.4 New training pipeline — `ai/train_ai.py` rewrite

- Step 1: Load annotated images from `ai/data/training/{doc_type}/`
- Step 2: Train YOLOv8 model on element detection
- Step 3: Generate element crops from annotated data
- Step 4: Train Siamese network on element verification (genuine vs tampered pairs)
- Step 5: Export models to `ai/models/weights/`
- Step 6: Generate reference embeddings for each element per document type → save to `ai/models/embeddings/{doc_type}/`

### 3.5 Training data preparation

- Create annotation tool workflow (use Roboflow or LabelImg)
- Annotate existing reference images
- Use `ai/augment_genuine.py` to generate augmented training data
- Generate synthetic forgeries (wrong logo, shifted positions, color changes, etc.)
- Store in `ai/data/training/{doc_type}/images/` and `ai/data/training/{doc_type}/labels/`

---

## Phase 4: Pipeline Restructuring

Rewrite `api/services/verification_orchestrator.py` and `api/services/verification_steps_service.py` for the new 4-layer pipeline:

### New Pipeline Stages (in order)

| Layer | Stage Name               | Input                                    | Output                           |
| ----- | ------------------------ | ---------------------------------------- | -------------------------------- |
| 0     | `DOCUMENT_IMAGE_QUALITY` | Raw uploaded image                       | Pass/fail (keep existing)        |
| 0     | `DOCUMENT_CROPPING`      | Raw image                                | Rectified/cropped document image |
| 1     | `FACE_EXTRACTION`        | Rectified document                       | Cropped face from document       |
| 1     | `FACE_MATCHING`          | Document face + selfie                   | Match score + pass/fail          |
| 2     | `OCR_EXTRACTION`         | Rectified document                       | Extracted text (raw)             |
| 3     | `AI_VERIFICATION`        | Rectified document + OCR text + doc type | Per-element verification results |
| 3     | `DATA_VERIFICATION`      | OCR extracted data                       | Check against citizen_records DB |
| 4     | `BLOCKCHAIN_RECORD`      | Document hash + metadata                 | MultiChain TX + IPFS CID         |

### Layer 1 — Biometric

- Extract face from document (using YOLO detection or ROI coordinates)
- Compare document face vs selfie using DeepFace (Facenet)
- Return match score and pass/fail

### Layer 2 — OCR

- Send rectified document to Google Vision API
- Return raw extracted text only (no verification)
- Pass text forward to Layer 3

### Layer 3 — AI

- Run YOLOv8 element detection on document
- Run Siamese verification on each element
- Validate positions, sizes, colors, fonts
- Check for ghost image, secondary logos, background patterns
- After all element checks pass → call DB verification:
  - Parse OCR text to extract structured fields (national ID, name, DOB, etc.)
  - Query `citizen_records` by national_id
  - If not found → create new record → proceed
  - If found and matches → proceed
  - If found and doesn't match → return FRAUD

### Layer 4 — Blockchain

- Compute SHA-256 hash of document
- Check for duplicates in `document_hashes` table
- Pin document to IPFS via Kubo API
- Publish metadata to MultiChain stream
- Store hash + CID + TX in `document_hashes` table
- Return all blockchain-related data

---

## Phase 5: Flutter App Changes

### 5.1 Document Capture Enhancement — `app/lib/screens/camera/document_capture_screen.dart`

- The frame rectangle overlay already exists (`_DocumentFrameOverlay`)
- **Add**: Auto-crop the captured image to the rectangle bounds before saving
- Use the overlay rectangle coordinates relative to the preview to crop the image
- Ensure the document fills the rectangle (optionally add edge-detection with `opencv_dart` package or simple coordinate-based cropping)

### 5.2 Loading/Progress State — `app/lib/screens/verification/verification_result_screen.dart`

- Already has step-by-step progress display with polling — verify it works with the new 8-stage pipeline
- Update stage names/icons to match new pipeline stages
- Ensure clear Arabic labels for each stage

### 5.3 Remove Fabric/Ledger References

- Remove `LedgerService` from Flutter services
- Update `VerificationResultScreen` to not reference Fabric-specific data

### 5.4 Update Verification Result Display

- Show per-element AI verification details (logo ✅ 96%, stamp ✅ 92%, etc.)
- Show OCR extracted text
- Show database verification result
- Show blockchain CID and TX ID

---

## Phase 6: Dashboard Changes

### 6.1 Admin Verification List Page — Create `dashboard/src/app/dashboard/verifications/page.tsx`

- Table with columns: ID, User Name, Document Type, Status, Date, Actions (View)
- **Filters** (all possible):
  - Status dropdown (pending / processing / verified / rejected)
  - Document type dropdown (from API)
  - User search (by name or email)
  - Date range picker (from — to)
  - Sort by (date, status, user) + ascending/descending
- Pagination with page size selector
- Click row → navigate to verification detail page

### 6.2 Verification Detail Page — Create `dashboard/src/app/dashboard/verifications/[id]/page.tsx`

- Show all verification metadata (user, document type, status, timestamps)
- Show pipeline steps with status + result details for each stage
- Show uploaded document image
- Show face match result (score + images)
- Show AI element-by-element results (table with element name, score, status)
- Show OCR extracted text
- Show DB verification result
- Show blockchain info (CID, TX, hash)
- **Admin Notes section**: list existing notes + textarea to add new note

### 6.3 Admin Notes API

- `POST /api/admin/verifications/{id}/notes` — add note (admin only)
- `GET /api/admin/verifications/{id}/notes` — list notes
- Create corresponding dashboard API proxy route

### 6.4 Admin Verification Filters API

- Update `GET /api/admin/verifications` to accept: `status`, `document_type_id`, `user_id`, `date_from`, `date_to`, `search`, `sort_by`, `sort_order`, `page`, `page_size`

### 6.5 Analytics Dashboard Overhaul — Rewrite `dashboard/src/app/dashboard/page.tsx`

- **Charting library**: Use **Tremor** (built for dashboards, Tailwind-compatible, professional look)
- **Summary Cards** (top row):
  - Total Users | Total Documents Verified | Success Rate | Avg Processing Time
- **Charts**:
  - Verifications over time (line chart — daily for last 30 days)
  - Verification status breakdown (donut/pie chart)
  - Verifications by document type (bar chart)
  - Success/failure rate trend (area chart)
  - Top failure reasons (horizontal bar chart)
  - User activity (verifications per user — top 10)
- **Export**: Button to export analytics as PDF or Excel

### 6.6 Analytics API Expansion

- Update `GET /api/admin/analytics` in `api/routers/admin_router.py`:
  - Add `date_from`, `date_to` query parameters
  - Return: summary counts, verifications_by_day (time series), verifications_by_type, verifications_by_status, failure_reasons, top_users, avg_processing_time
- Add `GET /api/admin/analytics/export?format=pdf|excel` endpoint

### 6.7 Advanced Reports Page — Create `dashboard/src/app/dashboard/reports/page.tsx`

- Document type analysis (most verified types, success rate per type)
- Fraud attempt patterns (failed verifications by failure stage, common failure reasons)
- User activity report (verifications per user, new registrations over time)
- System performance (average time per pipeline stage)
- Export all reports as PDF/Excel

### 6.8 Usability Improvements

- Consistent Arabic UI where applicable (button labels, table headers)
- Responsive design for all new pages
- Loading skeletons during data fetch
- Toast notifications for actions (Sonner already in place)
- Breadcrumb navigation for nested pages (verifications → detail)

---

## Phase 7: Blockchain Simplification

### 7.1 Consolidate to MultiChain only

- The `blockchain_verify` step in the pipeline should:
  1. Compute SHA-256 hash
  2. Check `document_hashes` for duplicate
  3. Pin file to IPFS (Kubo API — keep existing)
  4. Publish metadata to MultiChain stream `documents` (existing `multichain_service.py`)
  5. Store hash + IPFS CID + MultiChain TXID in `document_hashes` table
- Remove all Fabric references

### 7.2 Complete blockchain scenario

- Add verification endpoint: `GET /api/blockchain/verify/{hash}` — re-hash a document and compare against stored blockchain record to prove no tampering
- Store more complete metadata on MultiChain: document_type, verification_id, user_id, hash, ipfs_cid, timestamp, AI_decision

---

## Phase 8: Account Deactivation (Mark as Complete)

Current state from research:

- `PUT /api/v1/admin/users/{id}/suspend` — sets `is_active = False` ✅
- `PUT /api/v1/admin/users/{id}/activate` — sets `is_active = True` ✅
- `DELETE /api/v1/admin/users/{id}` — soft delete ✅
- Login blocks suspended users ✅
- Dashboard has suspend/activate buttons ✅

This is already functional — mark as ready in the report.

---

## Phase 9: Code Comments & Maintainability

- Add Arabic + English docstrings to all Python functions in api/, ai/, ocr/, Biometric/
- Add JSDoc comments to Dashboard components and API routes
- Add Dart documentation comments to Flutter screens and services

---

## Phase 10: DB Backup

- Create `scripts/backup_db.py` (or `.bat` for Windows):
  - Uses `mysqldump` to export full database
  - Saves to `backups/watheq_backup_{date}.sql`
  - Keeps last 7 backups (auto-cleanup)
- Create Windows Task Scheduler config or cron instruction for 24-hour interval

---

## Phase 11: Update Audit Report

Rewrite `REQUIREMENTS_AUDIT_REPORT.md` to reflect:

- **Removed requirements**: 1.4 (password recovery), 1.6 (profile update), 1.7 (sessions), 5.2.3 (re-verify), marked as "REMOVED — out of scope"
- **Deferred requirements**: Section 6 (notifications), marked as "DEFERRED"
- **Updated statuses**: All requirements adjusted per new decisions
- **New requirements added**: citizen DB verification, admin notes, advanced analytics, document auto-crop, admin verification filters
- **NFRs**: Simplified per user decisions (security relaxed, iOS ignored, scalability accepted, etc.)
- **Recalculated summary statistics**

---

## Execution Order

1. **Phase 1**: Remove Hyperledger Fabric, redundant biometric stage, server-side liveness — clean up codebase
2. **Phase 2**: Add `citizen_records` + `verification_notes` tables, expand verification filters in DB layer, seed mock data
3. **Phase 3**: Build YOLOv8 detection model + Siamese verification network, rewrite `verify_document.py` and `train_ai.py`
4. **Phase 4**: Restructure pipeline to 4 layers: Biometric → OCR → AI+DB → Blockchain (MultiChain only)
5. **Phase 5**: Flutter: document auto-crop, new pipeline stages UI, remove Fabric references
6. **Phase 6**: Dashboard: verification list page + filters + detail page + notes, analytics charts (Tremor), exports, advanced reports
7. **Phase 7**: Blockchain: consolidate to MultiChain, add verify endpoint, complete scenario
8. **Phase 8**: Verify account deactivation works end-to-end
9. **Phase 9**: Add code comments/docstrings throughout
10. **Phase 10**: Database backup script + scheduler config
11. **Phase 11**: Rewrite the audit report with all changes

---

## Verification Checklist

- [ ] Run full pipeline end-to-end: upload document → biometric → OCR → AI → blockchain → view result
- [ ] Test admin dashboard: verification list with all filters, detail page, add notes, analytics charts, export PDF/Excel
- [ ] Test citizen DB verification: new citizen → stored, existing match → pass, existing mismatch → fraud
- [ ] Test blockchain: verify document hash matches stored record
- [ ] Test document capture: image cropped to frame rectangle
- [ ] Verify DB backup script runs and produces valid SQL dump

---

## Key Decisions

| Decision                       | Choice                   | Rationale                                                                                                |
| ------------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| AI Model                       | YOLOv8 + Siamese Network | Per-element detection and verification — professional fraud detection approach                           |
| Blockchain                     | MultiChain only          | Hyperledger Fabric removed — simplifies architecture, avoids WSL dependency                              |
| Charting                       | Tremor                   | Tailwind-native, professional dashboard aesthetics                                                       |
| Doc Cropping                   | Client-side in Flutter   | Faster upload, immediate user feedback                                                                   |
| Citizen DB                     | Full record              | national_id, name_ar, name_en, DOB, address, issue_date, expiry_date, gender, nationality, document_type |
| Server-side Liveness           | Removed                  | Flutter ML Kit liveness is sufficient                                                                    |
| Biometric Pipeline Stage       | Deduplicated             | Single face matching step instead of two identical calls                                                 |
| Notifications                  | Deferred                 | Not needed for graduation project                                                                        |
| Security (CSRF, rate limiting) | Relaxed                  | University project scope                                                                                 |
| Unit Tests                     | Not needed               | Per user decision                                                                                        |
| iOS                            | Not targeted             | Android only                                                                                             |
