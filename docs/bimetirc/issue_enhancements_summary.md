# Issue Enhancements Summary

## Overview

Enhanced the Watheq project provisioning script to address missing biometric verification requirements and add more detailed issue definitions. The script now operates in **UPDATE MODE** - it will update existing issues instead of recreating them.

## Key Changes

### 1. **UPDATE Mode Implementation** ✅

- Added `update_issue()` function using GitHub REST API (PATCH endpoint)
- Modified `ensure_issue()` to detect existing issues by title and update them
- Preserves issue numbers, comments, and history
- Only creates new issues if they don't exist

### 2. **New Face Verification Issues** (3 new issues) ✅

#### Issue: "Face Verification Service Implementation"

- **Track:** AI/OCR
- **Size:** Medium
- **Milestone:** M2: Core Feature Delivery
- **Assignee:** toom20y
- **Details:**
  - Implement `backend/services/face_service.py` with face_recognition library
  - Create `/api/face-verify` FastAPI endpoint
  - Configurable threshold (default 0.6) with distance metrics
  - Comprehensive pytest suite with edge cases
  - Face detection preprocessing and alignment
  - Threshold tuning methodology documentation
  - **Acceptance:** >=90% accuracy on test dataset, graceful error handling

#### Issue: "Mobile Selfie Capture & Face Verification UX"

- **Track:** Frontend-Mobile
- **Size:** Medium
- **Milestone:** M2: Core Feature Delivery
- **Assignee:** AbobakrMahdii
- **Details:**
  - Camera integration with real-time face detection feedback
  - Selfie review screen with quality indicators
  - Offline queue with background sync
  - Integration with `/api/face-verify` endpoint
  - Accessibility features (voice guidance, high contrast)
  - Privacy controls (local encryption, automatic deletion)
  - Bilingual UI (Arabic/English) with culturally appropriate instructions

#### Issue: "Face Verification Dataset & Quality Assurance"

- **Track:** AI/OCR
- **Size:** Small
- **Milestone:** M2: Core Feature Delivery
- **Assignee:** toom20y
- **Details:**
  - Curate 20+ face pairs (50% matching, 50% non-matching)
  - Face cropping utility for consistency
  - Threshold calibration with ROC curve analysis
  - Basic liveness check exploration
  - Ethical guidelines and data retention policy

### 3. **Enhanced Existing Issues** (more detailed scope) ✅

#### "FastAPI Backend Architecture & Database Schema"

**New scope items added:**

- Complete database schema details (all table columns specified)
- Dependency injection pattern with abstract service interfaces
- Concrete adapter implementations for IPFS/Fabric
- FastAPI middleware (logging, correlation IDs, CORS)
- Pydantic schemas with comprehensive examples
- SQLAlchemy session management with connection pooling
- Health check endpoints (`/health`, `/readiness`)
- pytest-cov with 70% minimum coverage
- Structured logging with JSON output
- Docker-compose for PostgreSQL + Redis
- Test fixtures in conftest.py

#### "End-to-End Pipeline Integration & Migration"

**New scope items added:**

- Multi-step workflow breakdown (RECEIVED → PROCESSING → ... → COMPLETED)
- Detailed IPFS integration (pin_file, CID capture, retry queue)
- Fabric chaincode integration (RecordDocument invocation, transaction ID)
- Status state machine with substatus tracking
- Retry/compensation layer with exponential backoff (1m, 5m, 15m, 1h)
- CLI reconciliation tool (`scripts/retry_failed_operations.py`)
- Monitoring with correlation IDs and timing metrics
- Comprehensive error handling with rollback
- Idempotency checks to prevent duplicate ledger entries
- Transaction boundaries for atomicity
- Performance optimization with parallel processing (asyncio.gather)
- Timing breakdown: OCR 2-5s, forgery 1-3s, face 1-2s, IPFS 0.5-2s, Fabric 1-3s

## Total Issue Count

- **Before:** 17 issues
- **After:** 20 issues
- **New Issues:** 3 (all face verification related)
- **Enhanced Issues:** 2 (FastAPI Backend, Pipeline Integration)

## Alignment with Documentation

All changes align with **Watheq Academic Execution Plan** sections:

- **Section 3.3:** Face Verification requirements (face_recognition library, 0.6 threshold, liveness checks)
- **Section 3.4:** Mobile UX for selfie capture
- **Section 3.5-3.6:** End-to-end pipeline with IPFS and Fabric integration

## Running the Updated Script

### Option 1: Dry Run (Preview Changes)

```bash
python scripts/setup_watheq_project.py --dry-run
```

### Option 2: Update Existing Issues

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_token_here"

# Run the script (will UPDATE existing issues, not recreate)
python scripts/setup_watheq_project.py
```

## What Happens When You Run It

1. ✅ **Detects existing project** "Watheq Delivery Board" (reuses it)
2. ✅ **Updates existing issues** with enhanced content (preserves issue numbers)
3. ✅ **Creates 3 new issues** for face verification
4. ✅ **No duplicates** - existing issues updated in-place
5. ✅ **Preserves project structure** - no re-initialization

## Output Example

```
✓ Updated issue #12: FastAPI Backend Architecture & Database Schema
✓ Updated issue #13: End-to-End Pipeline Integration & Migration
✓ Created issue #21: Face Verification Service Implementation
✓ Created issue #22: Mobile Selfie Capture & Face Verification UX
✓ Created issue #23: Face Verification Dataset & Quality Assurance
```

## Benefits

1. **More Detailed Issues:** 2-5x more subtasks per major issue
2. **Complete Biometric Coverage:** All face verification requirements from docs now have dedicated issues
3. **Preserves History:** Updates don't lose issue numbers, comments, or progress
4. **Safe to Rerun:** Script detects existing resources and updates them

## Next Steps

1. Run the script to apply updates
2. Review enhanced issues in GitHub
3. Assign face verification issues to appropriate team members
4. Update project board to reflect new subtasks
5. Begin implementation following the detailed scope items
