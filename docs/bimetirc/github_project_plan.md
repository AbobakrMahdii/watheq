# Watheq GitHub Project Implementation Plan

This plan consolidates the requirements from `docs/Watheq_Academic_Execution_Plan_EN.md`, `docs/aggrement.md`, and the Arabic dossier to orchestrate a GitHub Project, milestones, and issues for the public `watheq` repository. Follow these steps before running any automation.

## 1. Prerequisites

- Personal access token (classic) with `project`, `repo`, and `workflow` scopes (`ghp_…`). Keep it secret and never commit it.
- GitHub GraphQL endpoint: `https://api.github.com/graphql`.
- Recommended CLI helpers: `gh` CLI or `curl` + `jq`.
- Repository slug: `abobakrmm/watheq` (replace owner if different).
- Time horizon: 12-week academic schedule (see Execution Plan).

### GraphQL request template

```bash
curl -s -H "Authorization: bearer $GITHUB_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query":"<GraphQL query string>","variables":{}}' \
     https://api.github.com/graphql | jq
```

Replace `<GraphQL query string>` with the mutations below (escape quotes/newlines as needed).

## 2. Create ProjectV2 board ("Watheq Delivery Board")

1. Fetch repository node ID:

   ```graphql
   query RepoId {
     repository(name: "watheq", owner: "abobakrmm") {
       id
     }
   }
   ```

2. Create a closed-scope project (repository-level):

   ```graphql
   mutation CreateProject($repoId: ID!) {
     createProjectV2(
       input: {
         ownerId: $repoId
         title: "Watheq Delivery Board"
         shortDescription: "Academic implementation tracker for the Watheq verification system"
       }
     ) {
       projectV2 {
         id
         number
         title
       }
     }
   }
   ```

3. Add custom fields via `projectV2AddField`:

   - `Status` (single-select): Backlog, In Progress, In Review, Blocked, Done.
   - `Priority` (single-select): Critical, High, Medium, Low.
   - `Track` (single-select): Governance, Backend, Frontend-Web, Frontend-Mobile, AI/OCR, Trust-Ledger, QA-Docs.
   - `Size` (single-select): Small, Medium, Large.

   Example mutation:

   ```graphql
   mutation AddSingleSelect($projectId: ID!) {
     projectV2AddField(
       input: {
         projectId: $projectId
         dataType: SINGLE_SELECT
         name: "Status"
         singleSelectOptions: [
           { name: "Backlog", color: BLUE }
           { name: "In Progress", color: YELLOW }
           { name: "In Review", color: ORANGE }
           { name: "Blocked", color: RED }
           { name: "Done", color: GREEN }
         ]
       }
     ) {
       field {
         ... on ProjectV2SingleSelectField {
           id
           name
         }
       }
     }
   }
   ```

   Repeat for the additional fields.

4. Configure a built-in `Iteration` field (optional) to mirror the weekly rhythm (`projectV2UpdateProjectV2ItemFieldValue`).
5. Enable views (`projectV2UpdateView` via REST or configure in UI) such as Kanban, Timeline, Issue table.

## 3. Milestones (Repository-level)

Create milestones capturing the semester flow. Use `createMilestone` mutation.

| Milestone                             | Target (suggested) | Summary                                                            |
| ------------------------------------- | ------------------ | ------------------------------------------------------------------ |
| **M1: Foundation & Setup**            | Week 2             | Repository hygiene, environment prep, baseline docs.               |
| **M2: Core Feature Delivery**         | Week 6             | OCR, forgery, face match, web/mobile scaffolds, FastAPI endpoints. |
| **M3: Trust & Integration Migration** | Week 8             | IPFS/Fabric services, document hashing, end-to-end orchestration.  |
| **M4: QA, Demo, and Handover**        | Week 10            | Testing battery, bilingual docs, presentation assets.              |

Mutation template:

```graphql
mutation CreateMilestone($repoId: ID!) {
  createMilestone(
    input: {
      repositoryId: $repoId
      title: "M3: Trust & Integration Migration"
      description: "Consolidate IPFS, blockchain logging, and full pipeline integration per academic plan."
      dueOn: "2026-01-15T00:00:00Z"
    }
  ) {
    milestone {
      id
      title
      number
      dueOn
    }
  }
}
```

Adjust `dueOn` dates to match your academic calendar.

## 4. Label Taxonomy

Use REST or GraphQL to ensure the following labels exist (REST example: `POST /repos/{owner}/{repo}/labels`).

- Size: `size:large`, `size:medium`, `size:small`.
- Importance: `importance:required`, `importance:bonus`, `importance:overkill`.
- Track: `track:governance`, `track:backend`, `track:frontend-web`, `track:frontend-mobile`, `track:ai-ocr`, `track:trust-ledger`, `track:qa-docs`.
- Status helpers (optional): `blocked`, `needs-docs`.

## 5. Issue Backlog Blueprint

Create issues with `createIssue`. Each issue body should follow this Markdown template:

```markdown
## Overview

- ...

## Scope

- [ ] Task 1
- [ ] Task 2

## Acceptance Criteria

- [ ] Criterion A

## Deliverables

- ...

## Resources

- ...

## Dependencies

- ...
```

Link issues to ProjectV2 using `addProjectV2ItemById` and set project fields with `projectV2UpdateProjectV2ItemFieldValue`.

### 5.1 Abobakr (Team Lead)

1. **Issue:** `Project Governance and Branch Hygiene`

   - Labels: `size:medium`, `importance:required`, `track:governance`.
   - Milestone: M1.
   - Summary: Establish Notion hub, weekly rhythm, branch protection, PR template.
   - Key tasks: finalize collaboration hub, configure GitHub settings, document workflows.
   - Resources: GitHub docs on protected branches; Notion project templates.

2. **Issue:** `Next.js Admin Console Migration`

   - Labels: `size:large`, `importance:required`, `track:frontend-web`.
   - Milestone: M2.
   - Body: Replatform React/Vite plan into Next.js 14 (App Router), Tailwind design system, admin dashboards for verification pipeline.
   - Acceptance: role-based views, document detail, integration with FastAPI endpoints.
   - Resources: Next.js docs, TanStack Query, Shadcn UI.

3. **Issue:** `Flutter Client Blueprint and Handoff`

   - Labels: `size:medium`, `importance:required`, `track:frontend-mobile`.
   - Milestone: M2.
   - Scope: Flutter skeleton, internationalization (ar/en), secure storage, API client spec, camera capture flow.
   - Resources: Flutter localization guide, camera plugin docs.

4. **Issue:** `FastAPI Backend Architecture & Database Schema`

   - Labels: `size:large`, `importance:required`, `track:backend`.
   - Milestone: M2.
   - Scope: Database migration to PostgreSQL (from SQLite baseline), SQLAlchemy models mirroring plan tables, Alembic scripts, API skeleton.
   - Acceptance: Running dev server with CRUD for documents, orchestrator placeholder endpoints.
   - Resources: FastAPI SQLModel/Alembic guides, PostgreSQL docker-compose.

5. **Issue:** `End-to-End Pipeline Integration & Migration`

   - Labels: `size:large`, `importance:required`, `track:trust-ledger`.
   - Milestone: M3 (migration requirement).
   - Scope: Chain orchestrator hooking OCR, forgery, face verification, IPFS, Fabric; error handling; document status transitions.
   - Resources: Fabric SDK samples, IPFS HTTP API, Python retry patterns.

6. **Issue:** `Release Candidate Cutover & Demo Packaging`
   - Labels: `size:medium`, `importance:required`, `track:qa-docs`.
   - Milestone: M4.
   - Scope: Freeze branch, coordinate QA sign-off, compile bilingual user guide, prepare demo script and video.

### 5.2 Fatima (@toom20y) – OCR & Visual AI

1. **Issue:** `OCR Service Implementation with EasyOCR`

   - Labels: `size:medium`, `importance:required`, `track:ai-ocr`.
   - Milestone: M2.
   - Tasks: EasyOCR reader, preprocessing pipeline, FastAPI endpoint, unit tests.
   - Resources: EasyOCR docs, OpenCV tutorials.

2. **Issue:** `Document Forgery Detection Baseline`

   - Labels: `size:medium`, `importance:required`, `track:ai-ocr`.
   - Milestone: M2.
   - Tasks: SSIM scorer, edge-based heuristics, pytest, documentation of thresholds.

3. **Issue:** `Dataset Assembly & Annotation Toolkit`
   - Labels: `size:small`, `importance:bonus`, `track:ai-ocr`.
   - Milestone: M1.
   - Tasks: Organize raw vs processed datasets, annotation notebook, storage guidelines.

### 5.3 Yamamah (@yamam1d) – Blockchain & IPFS

1. **Issue:** `IPFS Infrastructure and Service Wrapper`

   - Labels: `size:medium`, `importance:required`, `track:trust-ledger`.
   - Milestone: M3.
   - Tasks: go-ipfs setup, pinning script, FastAPI integration contract.

2. **Issue:** `Hyperledger Fabric Test Network & Chaincode`

   - Labels: `size:large`, `importance:required`, `track:trust-ledger`.
   - Milestone: M3.
   - Tasks: Fabric test network automation, chaincode for document hashes, SDK integration examples.

3. **Issue:** `Ledger Reliability & Retry Strategy`
   - Labels: `size:small`, `importance:bonus`, `track:trust-ledger`.
   - Milestone: M3.
   - Tasks: Queue design, failure logging, replay script.

### 5.4 Unassigned Backlog (for @hala-2891, @ReemRashad…, @vbxtyu789)

Create but leave unassigned until roles are finalized.

1. `UX Research & Figma Wireframes` (size:medium, importance:required, track:frontend-web, Milestone M1).
2. `Quality Assurance Playbook & Test Automation` (size:medium, importance:required, track:qa-docs, Milestone M4).
3. `Academic Documentation Pack` (size:medium, importance:required, track:qa-docs, Milestone M4).
4. `Risk Register and Mitigation Log` (size:small, importance:bonus, track:governance, Milestone M1).
5. `Analytics & Reporting Dashboard` (size:small, importance:bonus, track:frontend-web, Milestone M2).

Provide detailed bodies following the shared template with explicit task checklists, acceptance criteria, and references back to the execution plan.

## 6. GraphQL Mutations for Issues

Example `createIssue` mutation:

```graphql
mutation CreateIssue($repoId: ID!) {
  createIssue(
    input: {
      repositoryId: $repoId
      title: "FastAPI Backend Architecture & Database Schema"
      body: "## Overview\n- Consolidate the backend per academic plan..."
      milestoneId: "<milestone-id>"
      labelIds: ["<label-id>", "<label-id>"]
      assigneeIds: ["<user-id>"]
    }
  ) {
    issue {
      id
      number
      url
      title
    }
  }
}
```

Retrieve label, milestone, and user node IDs beforehand (`repository { labels(first:50) { nodes { id name } } }`).

### Link issue to project

```graphql
mutation AddToProject($projectId: ID!, $contentId: ID!) {
  addProjectV2ItemById(
    input: { projectId: $projectId, contentId: $contentId }
  ) {
    item {
      id
    }
  }
}
```

Then set field values:

```graphql
mutation SetField(
  $projectId: ID!
  $itemId: ID!
  $fieldId: ID!
  $optionId: String!
) {
  projectV2UpdateProjectV2ItemFieldValue(
    input: {
      projectId: $projectId
      itemId: $itemId
      fieldId: $fieldId
      value: { singleSelectOptionId: $optionId }
    }
  ) {
    projectV2Item {
      id
    }
  }
}
```

Repeat for Status, Priority, Track, Size. Store option IDs after creating the fields.

## 7. Automation & Quality Gates

- Configure required reviews on `main` (Settings → Branches).
- Add Issue template (`.github/ISSUE_TEMPLATE/implementation.md`) matching the body structure.
- Create GitHub Actions workflows for lint/test gates referencing FastAPI, Next.js, Flutter jobs.
- Weekly triage: update project fields, ensure blockers flagged, maintain due dates.

## 8. Resource Library

- FastAPI: <https://fastapi.tiangolo.com/>
- PostgreSQL Docker: <https://hub.docker.com/_/postgres>
- EasyOCR: <https://github.com/JaidedAI/EasyOCR>
- OpenCV Tutorials: <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html>
- face_recognition: <https://github.com/ageitgey/face_recognition>
- IPFS Docs: <https://docs.ipfs.tech/>
- Hyperledger Fabric Samples: <https://github.com/hyperledger/fabric-samples>
- Next.js App Router: <https://nextjs.org/docs/app>
- Flutter Internationalization: <https://docs.flutter.dev/development/accessibility-and-localization/internationalization>
- GitHub GraphQL API: <https://docs.github.com/en/graphql>
- Alembic migrations: <https://alembic.sqlalchemy.org/>

Review this plan, adjust dates/owners, and only then execute the mutations to populate the repository.
