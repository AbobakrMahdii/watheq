# Quick Reference: Running the Enhanced Setup Script

## What Changed

✅ **Script now updates existing issues instead of recreating them**
✅ **Added 3 new face verification issues**
✅ **Enhanced 2 existing issues with more detailed scope**

## How to Run

### Step 1: Verify your GitHub token is set

```bash
echo $GITHUB_TOKEN
```

If not set:

```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

### Step 2: Run the script

```bash
cd /c/Users/zabob/Desktop/UN/project
python scripts/setup_watheq_project.py
```

## What the Script Will Do

1. **Reuse existing project** "Watheq Delivery Board" (no duplicate)
2. **Update 17 existing issues** with current or enhanced content
3. **Create 3 NEW issues** for face verification:
   - Face Verification Service Implementation
   - Mobile Selfie Capture & Face Verification UX
   - Face Verification Dataset & Quality Assurance

## Expected Output

```
Existing project: Watheq Delivery Board (reusing)
✓ Updated issue #8: Project Governance and Branch Hygiene
✓ Updated issue #9: Next.js Admin Console Migration
✓ Updated issue #10: Flutter Client Blueprint and Handoff
✓ Updated issue #11: FastAPI Backend Architecture & Database Schema
✓ Updated issue #12: End-to-End Pipeline Integration & Migration
... (13 more updates)
✓ Created issue #24: Face Verification Service Implementation
✓ Created issue #25: Mobile Selfie Capture & Face Verification UX
✓ Created issue #26: Face Verification Dataset & Quality Assurance

Provisioning complete. Review GitHub project, milestones, and issues for accuracy.
```

## Verify Results

After running, check:

- GitHub Issues: https://github.com/AbobakrMahdii/watheq/issues
- Project Board: https://github.com/users/AbobakrMahdii/projects/17

You should see:

- ✅ 20 total issues (17 updated + 3 new)
- ✅ Enhanced issues have more detailed scope items
- ✅ Face verification fully covered

## Safe to Rerun

The script is **idempotent** - you can run it multiple times:

- Existing issues will be updated to match definitions
- No duplicates will be created
- Project structure preserved

## Dry Run (Optional)

To preview changes without making them:

```bash
python scripts/setup_watheq_project.py --dry-run
```
