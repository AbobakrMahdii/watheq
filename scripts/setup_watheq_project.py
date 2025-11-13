"""Provision the Watheq GitHub project structure via the GraphQL API.

This script creates (or idempotently reuses) a repository-scoped ProjectV2,
labels, milestones, and detailed issues that align with the academic execution
plan. It can run in `--dry-run` mode to print the operations without contacting
GitHub. When executing against GitHub, export `GITHUB_TOKEN` with a personal
access token that has `repo`, `project`, and `workflow` scopes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional

import requests

API_URL = "https://api.github.com/graphql"


@dataclass(frozen=True)
class IssueDefinition:
    title: str
    body: str
    labels: List[str]
    milestone: str
    assignee: Optional[str]
    project_fields: Dict[str, str]


class GitHubProvisioner:
    def __init__(
        self, token: str, owner: str, repo: str, dry_run: bool, verbose: bool
    ) -> None:
        self.token = token
        self.owner = owner
        self.repo = repo
        self.dry_run = dry_run
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
            }
        )

        # IDs populated at runtime.
        self.repo_id: Optional[str] = None
        self.owner_id: Optional[str] = None  # Organization or User ID
        self.project_id: Optional[str] = None
        self.project_number: Optional[int] = None
        self.project_fields: Dict[str, str] = {}
        self.project_field_options: Dict[str, Dict[str, str]] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        self.milestones: Dict[str, Dict[str, str]] = {}
        self.users: Dict[str, str] = {}
        self.existing_issues: Dict[str, Dict[str, str]] = {}
        self._dry_run_counter = 0

    # ------------------------------------------------------------------
    # GraphQL helpers
    # ------------------------------------------------------------------
    def _graphql(
        self, query: str, variables: Optional[Dict[str, Any]] = None, note: str = ""
    ) -> Dict[str, Any]:
        payload = {"query": query, "variables": variables or {}}
        if self.dry_run:
            self._dry_run_counter += 1
            header = (
                f"DRY-RUN {self._dry_run_counter}: {note}"
                if note
                else f"DRY-RUN {self._dry_run_counter}"
            )
            print(header)
            if self.verbose:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return {}

        # Log full request
        if os.getenv("DEBUG_GRAPHQL"):
            print(f"\n{'='*80}")
            print(f"GraphQL Request: {note or 'no note'}")
            print(f"{'='*80}")
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"{'='*80}\n")

        response = self.session.post(API_URL, data=json.dumps(payload))

        # Log full response
        if os.getenv("DEBUG_GRAPHQL"):
            print(f"\n{'='*80}")
            print(f"GraphQL Response: {note or 'no note'}")
            print(f"Status Code: {response.status_code}")
            print(f"{'='*80}")
            print(json.dumps(response.json(), indent=2, sort_keys=True))
            print(f"{'='*80}\n")

        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned {response.status_code}: {response.text}"
            )
        result = response.json()
        if "errors" in result:
            print(f"\n❌ Error in '{note}':")
            for error in result.get("errors", []):
                msg = error.get("message", "Unknown error")
                print(f"   {msg}")
            raise RuntimeError(
                f"GitHub API errors: {json.dumps(result['errors'], indent=2)}"
            )
        return result.get("data", {})

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def load_repository_state(self) -> None:
        if self.dry_run:
            self.repo_id = "DRYRUN-REPO-ID"
            self.owner_id = "DRYRUN-OWNER-ID"
            self.project_fields = {}
            self.labels = {}
            self.milestones = {}
            self.existing_issues = {}
            return

        query = dedent(
            """
            query RepoSnapshot($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) {
                id
                owner {
                  id
                  login
                  ... on User {
                    projectsV2(first: 50) {
                      nodes {
                        id
                        title
                        number
                        fields(first: 20) {
                          nodes {
                            ... on ProjectV2SingleSelectField {
                              id
                              name
                              options {
                                id
                                name
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  ... on Organization {
                    projectsV2(first: 50) {
                      nodes {
                        id
                        title
                        number
                        fields(first: 20) {
                          nodes {
                            ... on ProjectV2SingleSelectField {
                              id
                              name
                              options {
                                id
                                name
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                labels(first: 100) {
                  nodes {
                    id
                    name
                    color
                  }
                }
                milestones(first: 20, states: [OPEN]) {
                  nodes {
                    id
                    number
                    title
                    description
                    dueOn
                  }
                }
                issues(first: 100, states: [OPEN]) {
                  nodes {
                    id
                    number
                    title
                    url
                  }
                }
              }
            }
            """
        )
        data = self._graphql(
            query,
            {"owner": self.owner, "name": self.repo},
            note="Load repository state",
        )
        repo = data.get("repository") or {}
        if not repo:
            raise RuntimeError(f"Repository {self.owner}/{self.repo} not found")

        self.repo_id = repo["id"]

        # Extract owner ID (organization or user)
        owner_info = repo.get("owner") or {}
        self.owner_id = owner_info.get("id")
        if not self.owner_id:
            raise RuntimeError("Could not determine repository owner ID")

        self.labels = {
            node["name"]: node for node in repo.get("labels", {}).get("nodes", [])
        }
        self.milestones = {
            node["title"]: node for node in repo.get("milestones", {}).get("nodes", [])
        }
        self.existing_issues = {
            node["title"]: node for node in repo.get("issues", {}).get("nodes", [])
        }

        # Load projects from owner (User or Organization), not repository
        projects_list = owner_info.get("projectsV2", {}).get("nodes", [])
        for project in projects_list:
            if project["title"] == "Watheq Delivery Board":
                self.project_id = project["id"]
                self.project_number = project["number"]
                field_nodes = project.get("fields", {}).get("nodes", [])
                for field in field_nodes:
                    name = field.get("name")
                    if not name:
                        continue
                    self.project_fields[name] = field["id"]
                    self.project_field_options[name] = {
                        option["name"]: option["id"]
                        for option in field.get("options", [])
                    }
                print(
                    f"✓ Found existing project: Watheq Delivery Board #{self.project_number}"
                )
                break

    def load_user_ids(self, logins: List[str]) -> None:
        unresolved = [login for login in logins if login and login not in self.users]
        if self.dry_run:
            for login in unresolved:
                self.users[login] = f"DRYRUN-USER-ID-{login}"
            return

        if not unresolved:
            return

        query_parts = []
        variables = {}
        for idx, login in enumerate(unresolved):
            alias = f"u{idx}"
            variables[alias] = login
            query_parts.append(f"{alias}: user(login: ${alias}) {{ id login }}")
        query = f"query({', '.join(f'$u{idx}: String!' for idx in range(len(unresolved)))}) {{ {' '.join(query_parts)} }}"
        data = self._graphql(query, variables, note="Resolve user node IDs")
        for idx, login in enumerate(unresolved):
            alias = f"u{idx}"
            node = data.get(alias)
            if node is None:
                raise RuntimeError(f"User '{login}' not found or inaccessible")
            self.users[login] = node["id"]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def ensure_project(self) -> None:
        if self.project_id:
            print(
                f"✓ Using existing project: #{self.project_number} (ID: {self.project_id})"
            )
            return
        if self.owner_id is None:
            raise RuntimeError("Owner ID not loaded")

        print("Creating new ProjectV2: 'Watheq Delivery Board'")
        mutation = dedent(
            """
            mutation CreateProject($ownerId: ID!) {
              createProjectV2(input: {
                ownerId: $ownerId
                                title: "Watheq Delivery Board"
              }) {
                projectV2 {
                  id
                  number
                }
              }
            }
            """
        )
        try:
            data = self._graphql(
                mutation, {"ownerId": self.owner_id}, note="Create ProjectV2"
            )
        except RuntimeError as exc:
            message = str(exc)
            if "Something went wrong" in message:
                # GitHub occasionally returns transient 5xx errors. Refresh repository
                # state to detect whether the project was created despite the error and
                # surface a more actionable message if it was not.
                self.load_repository_state()
                if self.project_id:
                    return
                raise RuntimeError(
                    "GitHub returned an internal error while creating the project. "
                    "Wait a moment and retry the script; if the failure persists, "
                    "include the GitHub request ID from the original error when "
                    "contacting support."
                ) from exc
            raise
        if self.dry_run:
            self.project_id = "DRYRUN-PROJECT-ID"
            self.project_number = 0
            return
        node = data.get("createProjectV2", {}).get("projectV2")
        if not node:
            raise RuntimeError("Project creation failed")
        self.project_id = node["id"]
        self.project_number = node["number"]
        # Refresh fields map
        self.refresh_project_fields()

    def refresh_project_fields(self) -> None:
        if self.dry_run:
            return
        query = dedent(
            """
            query ProjectFields($id: ID!) {
              node(id: $id) {
                ... on ProjectV2 {
                  fields(first: 20) {
                    nodes {
                      ... on ProjectV2SingleSelectField {
                        id
                        name
                        options {
                          id
                          name
                        }
                      }
                    }
                  }
                }
              }
            }
            """
        )
        data = self._graphql(
            query, {"id": self.project_id}, note="Refresh project fields"
        )
        project = data.get("node") or {}
        field_nodes = project.get("fields", {}).get("nodes", [])
        self.project_fields = {}
        self.project_field_options = {}
        for field in field_nodes:
            name = field.get("name")
            if not name:
                continue
            self.project_fields[name] = field["id"]
            self.project_field_options[name] = {
                option["name"]: option["id"] for option in field.get("options", [])
            }

    def ensure_project_field(self, field_name: str, options: List[str]) -> None:
        if field_name in self.project_fields:
            existing_options = self.project_field_options.get(field_name, {})
            missing = [opt for opt in options if opt not in existing_options]
            if missing and not self.dry_run:
                # Update the field with all options (existing + new)
                print(f"Adding missing options to '{field_name}': {', '.join(missing)}")
                field_id = self.project_fields[field_name]

                # Combine existing and new options
                all_option_names = list(existing_options.keys()) + missing
                all_option_inputs = [
                    {"name": opt_name, "color": "GRAY", "description": opt_name}
                    for opt_name in all_option_names
                ]

                mutation = dedent(
                    """
                    mutation UpdateField($fieldId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
                      updateProjectV2Field(input: {
                        fieldId: $fieldId
                        name: $name
                        singleSelectOptions: $options
                      }) {
                        projectV2Field {
                          ... on ProjectV2SingleSelectField {
                            id
                            name
                            options {
                              id
                              name
                            }
                          }
                        }
                      }
                    }
                    """
                )

                self._graphql(
                    mutation,
                    {
                        "fieldId": field_id,
                        "name": field_name,
                        "options": all_option_inputs,
                    },
                    note=f"Update field '{field_name}' with new options",
                )
                # Reload fields to capture new option IDs
                self.refresh_project_fields()
            if missing and self.dry_run:
                # Populate placeholders for dry-run continuity.
                for opt in missing:
                    existing_options[opt] = f"DRYRUN-{field_name.upper()}-{opt.upper()}"
                self.project_field_options[field_name] = existing_options
            return

        if self.project_id is None:
            raise RuntimeError("Project not available")
        mutation = dedent(
            """
            mutation AddField($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
              createProjectV2Field(input: {
                projectId: $projectId
                dataType: SINGLE_SELECT
                name: $name
                singleSelectOptions: $options
              }) {
                projectV2Field {
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                  }
                }
              }
            }
            """
        )
        option_inputs = [
            {"name": opt, "color": "GRAY", "description": opt} for opt in options
        ]
        self._graphql(
            mutation,
            {
                "projectId": self.project_id,
                "name": field_name,
                "options": option_inputs,
            },
            note=f"Ensure project field {field_name}",
        )
        if self.dry_run:
            field_id = f"DRYRUN-FIELD-{field_name.upper().replace(' ', '-') }"
            self.project_fields[field_name] = field_id
            self.project_field_options[field_name] = {
                opt: f"DRYRUN-{field_name.upper().replace(' ', '-')}-{opt.upper().replace(' ', '-') }"
                for opt in options
            }
            return
        # Reload fields to capture option IDs
        self.refresh_project_fields()

    def ensure_labels(self, label_defs: Dict[str, Dict[str, str]]) -> None:
        for name, meta in label_defs.items():
            if name in self.labels:
                continue
            mutation = dedent(
                """
                mutation CreateLabel($repoId: ID!, $name: String!, $color: String!, $description: String) {
                  createLabel(input: {
                    repositoryId: $repoId
                    name: $name
                    color: $color
                    description: $description
                  }) {
                    label {
                      id
                      name
                    }
                  }
                }
                """
            )
            variables = {
                "repoId": self.repo_id,
                "name": name,
                "color": meta["color"].lstrip("#").upper(),
                "description": meta.get("description"),
            }
            data = self._graphql(mutation, variables, note=f"Create label {name}")
            if self.dry_run:
                self.labels[name] = {
                    "id": f"DRYRUN-LABEL-{name.upper().replace(':', '-')}",
                    "name": name,
                }
            else:
                node = data.get("createLabel", {}).get("label")
                if not node:
                    raise RuntimeError(f"Failed to create label {name}")
                self.labels[name] = node

    def ensure_milestones(self, milestone_defs: List[Dict[str, str]]) -> None:
        for meta in milestone_defs:
            title = meta["title"]
            if title in self.milestones:
                continue

            # Use REST API to create milestones (GraphQL doesn't support it)
            if self.dry_run:
                print(f"[DRY-RUN] Would create milestone: {title}")
                self.milestones[title] = {
                    "id": f"DRYRUN-MILESTONE-{title.upper().replace(' ', '-')}",
                    "title": title,
                }
                continue

            print(f"Creating milestone: {title}")
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/milestones"
            payload = {
                "title": title,
                "description": meta.get("description"),
            }
            if meta.get("due_on"):
                payload["due_on"] = meta["due_on"]

            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json=payload,
                timeout=30,
            )

            if response.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to create milestone {title}: {response.status_code} {response.text}"
                )

            milestone_data = response.json()
            # Convert REST API response to match our internal format
            self.milestones[title] = {
                "id": milestone_data[
                    "node_id"
                ],  # Use node_id for GraphQL compatibility
                "title": title,
            }

    def add_issue_to_project(
        self, issue_id: str, project_fields: Dict[str, str]
    ) -> None:
        if not self.project_id:
            return
        mutation_item = dedent(
            """
            mutation AddItem($projectId: ID!, $contentId: ID!) {
              addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
                item { id }
              }
            }
            """
        )
        data = self._graphql(
            mutation_item,
            {"projectId": self.project_id, "contentId": issue_id},
            note="Add issue to project",
        )
        item_id = None
        if self.dry_run:
            item_id = f"DRYRUN-ITEM-{issue_id}"
        else:
            item_id = data.get("addProjectV2ItemById", {}).get("item", {}).get("id")
            if not item_id:
                raise RuntimeError("Failed to add issue to project")

        for field_name, option_name in project_fields.items():
            field_id = self.project_fields.get(field_name)
            option_id = self.project_field_options.get(field_name, {}).get(option_name)
            if not field_id or not option_id:
                raise RuntimeError(
                    f"Project field or option missing for field '{field_name}' option '{option_name}'."
                )
            mutation_set = dedent(
                """
                mutation SetField($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
                  updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId
                    itemId: $itemId
                    fieldId: $fieldId
                    value: { singleSelectOptionId: $optionId }
                  }) {
                    projectV2Item { id }
                  }
                }
                """
            )
            self._graphql(
                mutation_set,
                {
                    "projectId": self.project_id,
                    "itemId": item_id,
                    "fieldId": field_id,
                    "optionId": option_id,
                },
                note=f"Set project field {field_name}={option_name}",
            )

    def update_issue(self, issue_number: int, definition: IssueDefinition) -> None:
        """Update an existing issue using REST API."""
        label_names = definition.labels
        milestone_title = definition.milestone
        milestone_number = self.milestones[milestone_title]["number"]

        # Use REST API to update issue
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}"
        payload = {
            "title": definition.title,
            "body": definition.body,
            "labels": label_names,
            "milestone": milestone_number,
        }
        if definition.assignee:
            payload["assignees"] = [definition.assignee]

        if self.dry_run:
            print(f"[DRY-RUN] Would update issue #{issue_number}: {definition.title}")
            return

        response = self.session.patch(url, json=payload)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to update issue #{issue_number}: {response.status_code} {response.text}"
            )
        print(f"✓ Updated issue #{issue_number}: {definition.title}")

    def ensure_issue(self, definition: IssueDefinition) -> None:
        if definition.title in self.existing_issues:
            issue_id = self.existing_issues[definition.title]["id"]
            issue_number = self.existing_issues[definition.title]["number"]

            # Update the issue content
            self.update_issue(issue_number, definition)

            # Update project fields if project exists
            if self.project_id:
                self.add_issue_to_project(issue_id, definition.project_fields)
            return

        label_ids = [self.labels[name]["id"] for name in definition.labels]
        milestone_id = self.milestones[definition.milestone]["id"]
        assignee_ids = []
        if definition.assignee:
            assignee_ids.append(self.users[definition.assignee])

        mutation = dedent(
            """
            mutation CreateIssue($input: CreateIssueInput!) {
              createIssue(input: $input) {
                issue {
                  id
                  number
                  title
                  url
                }
              }
            }
            """
        )
        issue_input = {
            "repositoryId": self.repo_id,
            "title": definition.title,
            "body": definition.body,
            "labelIds": label_ids,
            "milestoneId": milestone_id,
        }
        if assignee_ids:
            issue_input["assigneeIds"] = assignee_ids

        data = self._graphql(
            mutation, {"input": issue_input}, note=f"Create issue {definition.title}"
        )
        if self.dry_run:
            issue_id = f"DRYRUN-ISSUE-{definition.title.upper().replace(' ', '-') }"
        else:
            node = data.get("createIssue", {}).get("issue")
            if not node:
                raise RuntimeError(f"Failed to create issue {definition.title}")
            issue_id = node["id"]
            self.existing_issues[definition.title] = node

        if self.project_id:
            self.add_issue_to_project(issue_id, definition.project_fields)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        issues: List[IssueDefinition],
        label_defs: Dict[str, Dict[str, str]],
        milestone_defs: List[Dict[str, str]],
    ) -> None:
        self.load_repository_state()
        self.ensure_project()

        # Ensure project fields exist with known options.
        self.ensure_project_field(
            "Status", ["Backlog", "In Progress", "In Review", "Blocked", "Done"]
        )
        self.ensure_project_field("Priority", ["Critical", "High", "Medium", "Low"])
        self.ensure_project_field(
            "Track",
            [
                "Governance",
                "Backend",
                "Frontend-Web",
                "Frontend-Mobile",
                "AI/OCR",
                "Trust-Ledger",
                "QA-Docs",
            ],
        )
        self.ensure_project_field("Size", ["Small", "Medium", "Large"])

        self.ensure_labels(label_defs)
        self.ensure_milestones(milestone_defs)

        # Ensure we have user IDs for assignees.
        assignees = [issue.assignee for issue in issues if issue.assignee]
        self.load_user_ids([login for login in assignees if login])

        for issue in issues:
            self.ensure_issue(issue)


# ----------------------------------------------------------------------
# Static configuration
# ----------------------------------------------------------------------
LABEL_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "size:large": {
        "color": "5319E7",
        "description": "Large effort spanning multiple weeks",
    },
    "size:medium": {
        "color": "FBCA04",
        "description": "Medium effort with multiple sub-tasks",
    },
    "size:small": {
        "color": "0E8A16",
        "description": "Small effort that can wrap in under a week",
    },
    "importance:required": {
        "color": "B60205",
        "description": "Required for academic deliverable",
    },
    "importance:bonus": {"color": "1D76DB", "description": "Bonus stretch goal"},
    "importance:overkill": {
        "color": "5319E7",
        "description": "Intentionally beyond scope",
    },
    "track:governance": {
        "color": "A295D6",
        "description": "Leadership and coordination",
    },
    "track:backend": {"color": "0052CC", "description": "FastAPI and database"},
    "track:frontend-web": {"color": "5319E7", "description": "Web admin console"},
    "track:frontend-mobile": {"color": "FBCA04", "description": "Mobile client"},
    "track:ai-ocr": {"color": "D93F0B", "description": "OCR and forgery detection"},
    "track:trust-ledger": {"color": "0E8A16", "description": "IPFS and blockchain"},
    "track:qa-docs": {"color": "C2E0C6", "description": "Testing and documentation"},
    "blocked": {"color": "B60205", "description": "Blocked work item"},
    "needs-docs": {
        "color": "5319E7",
        "description": "Additional documentation required",
    },
}

MILESTONES: List[Dict[str, str]] = [
    {
        "title": "M1: Foundation & Setup",
        "description": "Repository hygiene, collaboration hub, baseline datasets, and tooling setup.",
        "due_on": "2025-11-30T00:00:00Z",
    },
    {
        "title": "M2: Core Feature Delivery",
        "description": "Deliver OCR, forgery detection, face verification, FastAPI scaffold, and client shells.",
        "due_on": "2025-12-28T00:00:00Z",
    },
    {
        "title": "M3: Trust & Integration Migration",
        "description": "Integrate IPFS, Fabric ledger, and orchestrate end-to-end submission flow (migration milestone).",
        "due_on": "2026-01-18T00:00:00Z",
    },
    {
        "title": "M4: QA, Demo, and Handover",
        "description": "QA hardening, bilingual documentation, demo assets, and release cutover.",
        "due_on": "2026-02-02T00:00:00Z",
    },
]


def build_issue_definitions() -> List[IssueDefinition]:
    def format_body(
        overview: List[str],
        scope: List[str],
        acceptance: List[str],
        deliverables: List[str],
        resources: List[str],
        dependencies: List[str],
    ) -> str:
        sections = ["## Overview\n" + "\n".join(f"- {line}" for line in overview)]
        sections.append("## Scope\n" + "\n".join(f"- [ ] {line}" for line in scope))
        sections.append(
            "## Acceptance Criteria\n"
            + "\n".join(f"- [ ] {line}" for line in acceptance)
        )
        sections.append(
            "## Deliverables\n" + "\n".join(f"- {line}" for line in deliverables)
        )
        sections.append("## Resources\n" + "\n".join(f"- {line}" for line in resources))
        sections.append(
            "## Dependencies\n"
            + "\n".join(f"- {line}" for line in dependencies or ["None"])
        )
        return "\n\n".join(sections)

    issues: List[IssueDefinition] = []

    issues.append(
        IssueDefinition(
            title="Project Governance and Branch Hygiene",
            labels=["size:medium", "importance:required", "track:governance"],
            milestone="M1: Foundation & Setup",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "Governance",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Stand up the governance layer so the Watheq team can execute the academic plan with predictable cadences.",
                    "Map repository protections, PR rituals, and documentation hubs to the contract obligations in docs/aggrement.md.",
                ],
                scope=[
                    "Configure Notion or Google Drive hub with backlog, meeting notes, and risk log templates.",
                    "Protect `main` with required reviews, status checks, and linear release tags per plan ",
                    "Draft and commit PR / issue templates aligning to academic reporting standards.",
                    "Publish weekly and mid-week sync cadence plus retro format in repo documentation.",
                ],
                acceptance=[
                    "Hub structure documented in README with owners and update frequency.",
                    "Branch protection rules active on GitHub and verified via screenshot / log.",
                    "Issue and PR templates live under .github and referenced by team.",
                    "Kickoff summary + risk register stored in collaboration hub and linked from project README.",
                ],
                deliverables=[
                    "Repository governance README section with branching + review policy.",
                    "Protected branch configuration evidence (export or screenshot).",
                    "Collaboration hub link with populated templates and first meeting record.",
                ],
                resources=[
                    "GitHub branch protection docs <https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/protecting-branches>",
                    "Notion project hub template inspiration <https://www.notion.so/templates/project-management>",
                    "Watheq contract summary in docs/aggrement.md",
                ],
                dependencies=["None"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Next.js Admin Console Migration",
            labels=["size:large", "importance:required", "track:frontend-web"],
            milestone="M2: Core Feature Delivery",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "Critical",
                "Track": "Frontend-Web",
                "Size": "Large",
            },
            body=format_body(
                overview=[
                    "Deliver a Next.js 14 admin console that visualises OCR, forgery, face-verification, and ledger metadata per execution plan section 3.4.",
                    "Ensure accessibility, bilingual readiness, and integration with the FastAPI orchestrator endpoints.",
                ],
                scope=[
                    "Bootstrap Next.js (App Router) with Tailwind design tokens mirroring mobile palette.",
                    "Implement authentication shell (mock SSO or email OTP) and role-based navigation for admins.",
                    "Build dashboard listing submitted documents with filters for status, ledger state, and verification scores.",
                    "Create document detail view showing OCR text, similarity metrics, face verification outcome, IPFS CID, Fabric transaction ID, and audit log.",
                    "Wire UI to backend endpoints using TanStack Query with optimistic updates and error toasts.",
                ],
                acceptance=[
                    "Admin can view, filter, and search submissions with live data from FastAPI dev server.",
                    "Detail view surfaces verification artefacts (text, scores, ledger metadata) with graceful loading states.",
                    "Components responsive (<=768px) and meet Arabic/English locale switching guidelines.",
                    "Unit tests or story-driven snapshot coverage for core pages (>=70% critical paths).",
                ],
                deliverables=[
                    "`frontend/` Next.js project with documented scripts for dev and build.",
                    "Design tokens and shared component primitives documented in README.",
                    "Integration notes capturing API contracts and sample payloads.",
                ],
                resources=[
                    "Next.js App Router docs <https://nextjs.org/docs/app>",
                    "TanStack Query patterns <https://tanstack.com/query/latest/docs/react/guides/queries>",
                    "Shadcn UI component recipes <https://ui.shadcn.com/docs>",
                    "Execution plan section 3.4 for required screens",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Flutter Client Blueprint and Handoff",
            labels=["size:medium", "importance:required", "track:frontend-mobile"],
            milestone="M2: Core Feature Delivery",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "Frontend-Mobile",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Establish a Flutter intake client that matches the academic UX expectations and provides a clean handoff for future collaborators.",
                    "Document submission workflow, bilingual layout, and API handshake to keep integration trivial later in the semester.",
                ],
                scope=[
                    "Spin up Flutter project with architecture folders (core, features, theming) and dependency injection scaffold.",
                    "Implement onboarding and authentication mock (email or student ID) with local secure storage for session token.",
                    "Create submission wizard: document capture (camera/gallery), selfie capture, metadata form, review screen, and submission confirmation.",
                    "Implement bilingual (ar/en) localization, RTL support, and theme parity with web dashboard.",
                    "Document API service layer interface, offline queue strategy, and error handling expectations.",
                ],
                acceptance=[
                    "App runs on emulator and physical device with stable navigation and localization toggles.",
                    "Submission flow stores draft locally when offline and retries gracefully.",
                    "README includes contributor handoff section with architecture overview and next steps.",
                    "Golden tests or widget tests cover the submission wizard states (minimum smoke coverage).",
                ],
                deliverables=[
                    "`mobile/` Flutter project with scripts for `flutter run` and `flutter test`.",
                    "Localization ARB files and documentation for adding new strings.",
                    "Architecture decision record describing queue/offline strategy.",
                ],
                resources=[
                    "Flutter localization guide <https://docs.flutter.dev/development/accessibility-and-localization/internationalization>",
                    "Flutter camera plugin <https://pub.dev/packages/camera>",
                    "Execution plan section 3.4 mobile requirements",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="FastAPI Backend Architecture & Database Schema",
            labels=["size:large", "importance:required", "track:backend"],
            milestone="M2: Core Feature Delivery",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "Critical",
                "Track": "Backend",
                "Size": "Large",
            },
            body=format_body(
                overview=[
                    "Implement the FastAPI backbone that mirrors the academic ERD and orchestrates OCR, forgery, face, IPFS, and Fabric services.",
                    "Migrate persistence to PostgreSQL with Alembic migrations, ensuring compatibility with future AI modules.",
                ],
                scope=[
                    "Docker-compose for PostgreSQL and local supporting services (Redis optional for retries).",
                    "SQLAlchemy models and Alembic migrations for documents, OCR results, verification scores, hash ledger metadata, and retry queue.",
                    "Define complete database schema: `documents` (id, user_name, document_type, upload_path, ipfs_cid, created_at), `ocr_results` (document_id, extracted_text, confidence), `verification_results` (document_id, seal_score, signature_score, face_score, overall_status), `document_hashes` (document_id, sha256, fabric_tx_id, ledger_status, recorded_at).",
                    "FastAPI routers for OCR, forgery, face verification, ledger lookup, and `/api/process-document` orchestration stub.",
                    "Implement dependency injection pattern with service interfaces: `OCRService`, `ForgeryService`, `FaceService`, `IPFSService`, `FabricService` as abstract base classes.",
                    "Create concrete service implementations with clear adapter pattern for external dependencies (IPFS client, Fabric SDK).",
                    "Add FastAPI middleware for request logging, correlation IDs, error handling, and CORS configuration.",
                    "Implement Pydantic schemas for request/response validation covering all API endpoints with examples.",
                    "Configure SQLAlchemy session management with connection pooling and transaction handling best practices.",
                    "Add health check endpoints (`/health`, `/readiness`) verifying database, IPFS, and Fabric connectivity.",
                    "pytest suite with database fixtures covering CRUD operations, service layer integration tests with mocks, and primary orchestrator happy-path.",
                    "Set up pytest-cov for code coverage reporting with minimum 70% threshold for critical paths.",
                    "Configure logging with structlog for JSON output, log levels per environment, and integration with monitoring tools.",
                ],
                acceptance=[
                    "`uvicorn` server boots with PostgreSQL backend and exposed OpenAPI docs.",
                    "Alembic upgrade/downgrade works cleanly and documented in README with migration workflow.",
                    "All service interfaces defined with clear contracts (methods, parameters, return types, exceptions).",
                    "Tests pass locally (`pytest -v`) and exercise service interfaces with mocks, achieving >=70% coverage.",
                    "Architecture diagram (C4 L2) checked into docs/ showing module boundaries and data flow.",
                    "Health check endpoints respond correctly when dependencies are up/down with appropriate status codes.",
                    "OpenAPI/Swagger UI accessible at `/docs` with comprehensive endpoint documentation and examples.",
                ],
                deliverables=[
                    "`backend/` FastAPI project with modular routers (api/routes/) and services (services/).",
                    "`backend/alembic/` migrations representing academic schema with version control.",
                    "Service interface documentation and dependency graph in docs/architecture/backend.md.",
                    "Docker-compose configuration for local development environment with PostgreSQL, Redis.",
                    "Test fixtures and utilities in `tests/conftest.py` for database setup/teardown.",
                ],
                resources=[
                    "FastAPI SQLModel/Alembic patterns <https://fastapi.tiangolo.com/tutorial/sql-databases/>",
                    "PostgreSQL Docker image <https://hub.docker.com/_/postgres>",
                    "Dependency injection in FastAPI <https://fastapi.tiangolo.com/tutorial/dependencies/>",
                    "SQLAlchemy best practices <https://docs.sqlalchemy.org/en/14/orm/session_basics.html>",
                    "Execution plan sections 3.1-3.6 and ERD references",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="End-to-End Pipeline Integration & Migration",
            labels=["size:large", "importance:required", "track:trust-ledger"],
            milestone="M3: Trust & Integration Migration",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "Critical",
                "Track": "Trust-Ledger",
                "Size": "Large",
            },
            body=format_body(
                overview=[
                    "Integrate OCR, forgery, face verification, IPFS, and Fabric into a resilient document processing pipeline (migration milestone).",
                    "Implement retry semantics and status transitions so submissions survive partial infrastructure outages.",
                ],
                scope=[
                    "Implement `/api/process-document` orchestrator with multi-step workflow: receive upload → persist file → extract OCR text → run forgery detection → perform face verification (if selfie provided) → aggregate results.",
                    "Integrate IPFS pinning: call `ipfs_service.pin_file()` to store document, capture CID, and handle connection failures with meaningful errors and retry queue.",
                    "Integrate Fabric chaincode: invoke `RecordDocument` with SHA-256 hash, CID, and metadata via Fabric SDK, capturing transaction ID and timestamp.",
                    "Implement status state machine: RECEIVED → PROCESSING → OCR_COMPLETE → VERIFICATION_COMPLETE → IPFS_PINNED → LEDGER_RECORDED → COMPLETED (or FAILED with substatus).",
                    "Add retry/compensation layer: create `retry_queue` table for failed IPFS pins and Fabric submissions, with background worker polling and exponential backoff (1m, 5m, 15m, 1h).",
                    "Build CLI reconciliation tool (`scripts/retry_failed_operations.py`) to manually trigger retries with filtering by operation type and age.",
                    "Expose monitoring/observability: add structured logging with correlation IDs per submission, timing metrics for each pipeline step, and failure classification (IPFS_UNREACHABLE, FABRIC_TIMEOUT, OCR_FAILED).",
                    "Create comprehensive error handling: wrap each service call in try-except with specific exception types, rollback database on critical failures, and return detailed error responses to clients.",
                    "Add idempotency: ensure resubmitting same document hash doesn't duplicate ledger entries, check existing records before processing.",
                    "Implement transaction boundaries: use database transactions to ensure atomicity of status updates and result persistence.",
                    "Add performance optimization: process OCR, forgery, and face verification in parallel where possible using asyncio.gather or ThreadPoolExecutor.",
                ],
                acceptance=[
                    "Sample CLI (`scripts/demo_workflow.py --doc sample.jpg --selfie selfie.jpg`) runs end-to-end producing CID, hash, and Fabric transaction ID with detailed step-by-step output.",
                    "Offline IPFS scenarios: daemon stopped during submission results in queued retry entry, restarting daemon allows successful reconciliation via retry script.",
                    "Offline Fabric scenarios: network unavailable during submission queues ledger write, manual replay via CLI script successfully records to chain when network restored.",
                    "Integration tests in `tests/test_pipeline.py` mock IPFS/Fabric but assert correct orchestrator branching, status transitions, and error handling for each failure mode.",
                    "At least one live smoke test documented in README demonstrating real IPFS + Fabric integration with all services running.",
                    "Migration runbook updated in `docs/runbook.md` with integration sequence, rollback strategy, and troubleshooting steps for common failures.",
                    "Monitoring logs demonstrate timing breakdown: OCR 2-5s, forgery 1-3s, face 1-2s, IPFS 0.5-2s, Fabric 1-3s per operation.",
                ],
                deliverables=[
                    "Integrated FastAPI service with complete `/api/process-document` orchestrator endpoint handling all workflow steps.",
                    "Retry queue implementation with background worker or manual replay script in `scripts/`.",
                    "Monitoring/log configuration with correlation IDs and metrics documented in docs/runbook.md.",
                    "Demo CLI script with comprehensive output and error handling demonstration.",
                    "Integration test suite covering happy path, partial failures, and complete outages.",
                ],
                resources=[
                    "IPFS HTTP client docs <https://docs.ipfs.tech/reference/http/>",
                    "Hyperledger Fabric Python SDK samples <https://github.com/hyperledger/fabric-sdk-py>",
                    "FastAPI background tasks <https://fastapi.tiangolo.com/tutorial/background-tasks/>",
                    "Python asyncio patterns <https://docs.python.org/3/library/asyncio-task.html>",
                    "Execution plan section 3.5-3.6 for trust services",
                ],
                dependencies=[
                    "FastAPI Backend Architecture & Database Schema",
                    "Hyperledger Fabric Test Network & Chaincode",
                ],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Release Candidate Cutover & Demo Packaging",
            labels=["size:medium", "importance:required", "track:qa-docs"],
            milestone="M4: QA, Demo, and Handover",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "QA-Docs",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Coordinate QA sign-off, produce bilingual documentation, and prepare the academic demo package for evaluation week.",
                    "Ensure migration plan concludes with tidy branch state, issue closure, and artifact distribution.",
                ],
                scope=[
                    "Lock release branch, merge final fixes, and tag RC build with changelog summarising milestones.",
                    "Run full test battery (backend, web, mobile) and archive reports/screenshots in docs/test_plan.md.",
                    "Compile bilingual user manual, operations runbook, and quickstart one-pager for graders.",
                    "Script 3-5 minute demo video plan, capture footage, and deliver packaged assets via shared drive/YouTube unlisted link.",
                ],
                acceptance=[
                    "Release branch tagged with RC version and matching release notes committed.",
                    "All required issues closed or referenced with rationale, board reflects Done state.",
                    "Documentation pack reviewed by at least one collaborator (peer sign-off recorded).",
                    "Demo assets hosted with access instructions in README.",
                ],
                deliverables=[
                    "Release checklist with sign-off timestamps.",
                    "docs/user_manual.md and docs/runbook.md finalised.",
                    "Demo video link and slide deck stored in repo or shared drive.",
                ],
                resources=[
                    "Execution plan Phase 4 guidance",
                    "GitHub release best practices <https://docs.github.com/repositories/releasing-projects-on-github/about-releases>",
                    "University presentation rubric (if provided)",
                ],
                dependencies=["End-to-End Pipeline Integration & Migration"],
            ),
        )
    )

    # Fatima (OCR & Visual AI)
    issues.append(
        IssueDefinition(
            title="OCR Service Implementation with EasyOCR",
            labels=["size:medium", "importance:required", "track:ai-ocr"],
            milestone="M2: Core Feature Delivery",
            assignee="toom20y",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "AI/OCR",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Deliver the OCR microservice baseline that extracts bilingual text from document uploads following execution plan section 3.1.",
                    "Provide deterministic outputs and persistence wiring so downstream verification modules can consume results.",
                ],
                scope=[
                    "Build `backend/services/ocr_service.py` with EasyOCR reader (ar/en) plus configurable preprocessing toggles.",
                    "Implement `/api/ocr` FastAPI route that stores uploads, invokes service, logs request metadata, and persists text/confidence.",
                    "Add preprocessing utilities (contrast, grayscale, denoise) gated by feature flags for tampering resilience.",
                    "Write pytest suite with real sample images and synthetic fixtures to guard against regressions.",
                ],
                acceptance=[
                    "Service returns text + confidence JSON for baseline documents with >=0.8 accuracy on provided dataset.",
                    "Endpoint handles multi-page PDF via pdf2image conversion with explicit tests.",
                    "Failures (unsupported format, unreadable text) bubble structured errors and are documented in README.",
                    "Code coverage instrumentation demonstrates exercised paths for preprocessing toggles.",
                ],
                deliverables=[
                    "OCR service module with docstrings and configuration notes.",
                    "FastAPI route + schema definitions documented in API reference.",
                    "Test report attached to issue or stored in docs/test_results/.",
                ],
                resources=[
                    "EasyOCR GitHub repo <https://github.com/JaidedAI/EasyOCR>",
                    "OpenCV preprocessing walkthrough <https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html>",
                    "Execution plan section 3.1 requirements",
                ],
                dependencies=["FastAPI Backend Architecture & Database Schema"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Document Forgery Detection Baseline",
            labels=["size:medium", "importance:required", "track:ai-ocr"],
            milestone="M2: Core Feature Delivery",
            assignee="toom20y",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "AI/OCR",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Implement structural similarity and edge-analysis checks to flag tampered seals/signatures per execution plan section 3.2.",
                    "Expose configurable thresholds and persist detailed metrics for downstream UI rendering.",
                ],
                scope=[
                    "Create `backend/services/forgery_service.py` implementing SSIM baseline with optional reference templates.",
                    "Design API `/api/forgery` for single and dual-image flows with threshold controls and outcome rationale.",
                    "Persist seal and signature scores in verification table, capturing heatmap artifacts when available.",
                    "Author synthetic tests comparing untouched vs. doctored samples, including threshold regression coverage.",
                ],
                acceptance=[
                    "Service returns score 0-1 with configurable threshold and textual explanation.",
                    "Edge-only fallback activates when no reference provided, with documented heuristics.",
                    "Heatmap artifacts optional but stored/linked for UI, or decision recorded when skipped.",
                    "Test suite demonstrates detection on at least five tampered exemplars (commit artifacts sanitized).",
                ],
                deliverables=[
                    "Forgery detection service + FastAPI route with schema docs.",
                    "Threshold rationale documented and tuning guidance shared with UI team.",
                    "Sample outputs archived under `data/processed/forgery/`.",
                ],
                resources=[
                    "scikit-image SSIM reference <https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity>",
                    "Canny edge detection tutorial <https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>",
                    "Execution plan section 3.2",
                ],
                dependencies=["OCR Service Implementation with EasyOCR"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Face Verification Service Implementation",
            labels=["size:medium", "importance:required", "track:ai-ocr"],
            milestone="M2: Core Feature Delivery",
            assignee="toom20y",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "AI/OCR",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Deliver biometric face verification service matching document photos with live selfies per execution plan section 3.3.",
                    "Implement face_recognition library with configurable thresholds and return similarity scores for downstream decision-making.",
                ],
                scope=[
                    "Build `backend/services/face_service.py` with `compare_face(document_face_path, selfie_path)` function using face_recognition library.",
                    "Implement face encoding extraction with error handling for no-face-detected scenarios.",
                    "Add configurable acceptance threshold (default 0.6) with distance metric normalization (0-1 scale).",
                    "Create `/api/face-verify` FastAPI route accepting two image uploads (document face + selfie) and returning match score + decision.",
                    "Persist verification results in `verification_results.face_score` column with metadata (distance, threshold used, timestamp).",
                    "Write pytest suite (`tests/test_face.py`) with sample face pairs (matching and non-matching) covering edge cases (no face, multiple faces, poor quality).",
                    "Add face detection preprocessing with alignment/cropping utilities for improved matching accuracy.",
                    "Document threshold tuning methodology and provide calibration dataset requirements.",
                ],
                acceptance=[
                    "Service successfully matches known face pairs with >=90% accuracy on test dataset.",
                    "API endpoint returns structured JSON with score, distance, decision, and confidence level.",
                    "Graceful error handling for missing faces, blurry images, or unsupported formats with meaningful error messages.",
                    "Tests cover at least 10 positive and 10 negative pairs with documented threshold sensitivity analysis.",
                    "README documents face_recognition library setup, dlib requirements, and troubleshooting common installation issues.",
                ],
                deliverables=[
                    "Face verification service module with comprehensive docstrings and type hints.",
                    "FastAPI route + Pydantic schemas documented in OpenAPI/Swagger.",
                    "Test report with accuracy metrics and confusion matrix stored in docs/test_results/face_verification.md.",
                    "Calibration guide for threshold tuning based on security vs. usability trade-offs.",
                ],
                resources=[
                    "face_recognition library <https://github.com/ageitgey/face_recognition>",
                    "dlib installation guide <http://dlib.net/compile.html>",
                    "Face verification best practices <https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/>",
                    "Execution plan section 3.3",
                ],
                dependencies=["FastAPI Backend Architecture & Database Schema"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Mobile Selfie Capture & Face Verification UX",
            labels=["size:medium", "importance:required", "track:frontend-mobile"],
            milestone="M2: Core Feature Delivery",
            assignee="AbobakrMahdii",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "Frontend-Mobile",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Integrate selfie capture workflow into mobile submission flow with real-time feedback and offline handling.",
                    "Ensure seamless UX for biometric verification aligned with document submission per execution plan section 3.3-3.4.",
                ],
                scope=[
                    "Implement camera integration using Flutter `camera` plugin or React Native Camera with in-app capture UI.",
                    "Add real-time face detection feedback during capture (face detected, center face, improve lighting) using ML Kit or equivalent.",
                    "Design selfie review screen allowing retake with quality indicators (brightness, face size, blur detection).",
                    "Build offline queue for selfie storage when network unavailable, with background sync on connectivity restoration.",
                    "Integrate with `/api/face-verify` endpoint, displaying verification results with clear pass/fail UI and retry options.",
                    "Implement accessibility features: voice guidance for capture, high contrast mode, and haptic feedback for capture confirmation.",
                    "Add privacy controls: local encryption for stored selfies, automatic deletion after successful submission, and clear data retention policy display.",
                    "Localize all UI strings for selfie capture flow (Arabic/English) with culturally appropriate instructions.",
                ],
                acceptance=[
                    "Camera opens with face detection overlay guiding user to optimal capture position.",
                    "Selfie successfully captured and stored locally with encryption, displayable in review screen.",
                    "Offline submissions queued and automatically retry face verification when online, with status notification.",
                    "Verification results display clearly with actionable feedback (accepted, rejected with reason, retry guidance).",
                    "Accessibility features tested with screen reader and high contrast mode enabled.",
                    "Privacy controls documented and displayed to users with consent flow before first capture.",
                ],
                deliverables=[
                    "Selfie capture screens integrated into mobile submission wizard with navigation flow.",
                    "Offline handling implementation with queue status visibility in app.",
                    "Privacy policy screen and consent mechanism implemented.",
                    "Widget/integration tests covering capture, offline storage, and verification result handling.",
                ],
                resources=[
                    "Flutter camera plugin <https://pub.dev/packages/camera>",
                    "ML Kit face detection <https://developers.google.com/ml-kit/vision/face-detection>",
                    "React Native Camera <https://github.com/react-native-camera/react-native-camera>",
                    "Mobile biometric UX patterns <https://material.io/design/platform-guidance/android-biometrics.html>",
                    "Execution plan sections 3.3-3.4",
                ],
                dependencies=[
                    "Flutter Client Blueprint and Handoff",
                    "Face Verification Service Implementation",
                ],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Face Verification Dataset & Quality Assurance",
            labels=["size:small", "importance:required", "track:ai-ocr"],
            milestone="M2: Core Feature Delivery",
            assignee="toom20y",
            project_fields={
                "Status": "Backlog",
                "Priority": "Medium",
                "Track": "AI/OCR",
                "Size": "Small",
            },
            body=format_body(
                overview=[
                    "Curate face verification dataset with document photo + selfie pairs and establish testing methodology for biometric accuracy.",
                    "Implement quality checks and threshold calibration supporting execution plan section 3.3 requirements.",
                ],
                scope=[
                    "Assemble dataset of 20+ face pairs: 10 matching (same person, document vs selfie) and 10 non-matching (different people).",
                    "Include challenging samples: varied lighting, angles, with/without glasses, different ages, diverse demographics.",
                    "Organize in `data/processed/face/` with clear naming convention (e.g., `person_01_doc.jpg`, `person_01_selfie.jpg`).",
                    "Create face cropping utility (`scripts/crop_faces.py`) to extract and normalize faces from documents for consistent comparison.",
                    "Implement threshold calibration script testing multiple acceptance thresholds (0.4-0.8) and reporting false accept/reject rates.",
                    "Add basic liveness check exploration: capture two selfies with head movement and compare for consistency.",
                    "Document dataset collection methodology, consent procedures, and ethical considerations in README.",
                    "Generate test report with ROC curve, optimal threshold recommendation, and accuracy metrics.",
                ],
                acceptance=[
                    "Dataset committed with at least 20 pairs (50% matching, 50% non-matching) covering diverse scenarios.",
                    "Cropping utility successfully extracts faces from documents with consistent output dimensions.",
                    "Calibration script produces threshold recommendation with supporting metrics (accuracy, precision, recall).",
                    "Liveness check concept validated with at least 3 test subjects and documented findings.",
                    "Ethical guidelines documented including anonymization strategy and data retention policy.",
                ],
                deliverables=[
                    "Face verification dataset with manifest documenting each pair and ground truth labels.",
                    "Face cropping script with usage documentation.",
                    "Calibration report stored in `docs/test_results/face_calibration.md` with ROC curve visualization.",
                    "Liveness check exploration notes and recommendations for future enhancement.",
                ],
                resources=[
                    "Face detection with dlib <http://dlib.net/face_landmark_detection.py.html>",
                    "OpenCV face detection tutorial <https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html>",
                    "Biometric testing methodology <https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt>",
                    "Execution plan section 3.3 bonus items",
                ],
                dependencies=[
                    "Dataset Assembly & Annotation Toolkit",
                    "Face Verification Service Implementation",
                ],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Dataset Assembly & Annotation Toolkit",
            labels=["size:small", "importance:bonus", "track:ai-ocr"],
            milestone="M1: Foundation & Setup",
            assignee="toom20y",
            project_fields={
                "Status": "Backlog",
                "Priority": "Medium",
                "Track": "AI/OCR",
                "Size": "Small",
            },
            body=format_body(
                overview=[
                    "Curate initial document dataset with tampered variants and create lightweight annotation helpers for OCR/forgery benchmarking.",
                ],
                scope=[
                    "Organize `data/raw` vs `data/processed` directories with consistent naming (doc_01_front.jpg, doc_01_selfie.jpg).",
                    "Produce 5+ tampered samples using controlled edits (seal removal, signature alteration) documented in log.",
                    "Draft Jupyter notebook or CLI to annotate bounding boxes for signatures/seals if reference-based detection desired.",
                    "Document dataset license considerations and backup strategy in README.",
                ],
                acceptance=[
                    "Dataset folders committed (sanitized) with README cataloging each sample and tamper description.",
                    "Annotation tool runs locally and exports JSON/CSV ready for ML tooling.",
                    "Storage and privacy guidance captured in documentation.",
                ],
                deliverables=[
                    "Structured dataset directories with sample manifest.",
                    "Annotation script/notebook plus usage documentation.",
                    "Dataset README including consent and ethical notes.",
                ],
                resources=[
                    "Execution plan Phase 2 data prep guidance",
                    "Label Studio inspiration <https://labelstud.io/>",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    # Yamamah (Blockchain & IPFS)
    issues.append(
        IssueDefinition(
            title="IPFS Infrastructure and Service Wrapper",
            labels=["size:medium", "importance:required", "track:trust-ledger"],
            milestone="M3: Trust & Integration Migration",
            assignee="yamam1d",
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "Trust-Ledger",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Provision local IPFS node and Python wrapper to pin documents, surface CIDs, and expose diagnostics per execution plan section 3.5.",
                ],
                scope=[
                    "Install and document go-ipfs setup, including daemon start/stop scripts and config tuned for local dev.",
                    "Implement `ledger/ipfs_service.py` with pin/unpin, retrieval, and health check functions.",
                    "Secure environment variable handling for IPFS endpoints and secrets (if any).",
                    "Add integration tests or CLI smoke test validating pin and retrieve cycle with sample file.",
                ],
                acceptance=[
                    "IPFS daemon startup documented with commands and expected logs.",
                    "Service wrapper returns CIDs and handles connection failures with meaningful errors.",
                    "Smoke test demonstrates storing document and retrieving content via CID.",
                    "Runbook updated with troubleshooting steps for common IPFS issues.",
                ],
                deliverables=[
                    "IPFS setup scripts and documentation under `infrastructure/`.",
                    "Python service wrapper with unit tests and logging hooks.",
                    "Smoke test script (CLI) verifying end-to-end pinning.",
                ],
                resources=[
                    "IPFS getting started guide <https://docs.ipfs.tech/install/>",
                    "ipfshttpclient usage reference <https://github.com/ipfs-shipyard/py-ipfs-http-client>",
                    "Execution plan section 3.5",
                ],
                dependencies=["FastAPI Backend Architecture & Database Schema"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Hyperledger Fabric Test Network & Chaincode",
            labels=["size:large", "importance:required", "track:trust-ledger"],
            milestone="M3: Trust & Integration Migration",
            assignee="yamam1d",
            project_fields={
                "Status": "Backlog",
                "Priority": "Critical",
                "Track": "Trust-Ledger",
                "Size": "Large",
            },
            body=format_body(
                overview=[
                    "Stand up Fabric test network and author chaincode to persist document hashes, tying into execution plan section 3.5.",
                    "Provide automation scripts so teammates can bootstrap network quickly during integration weeks.",
                ],
                scope=[
                    "Clone fabric-samples, automate `test-network` startup with two orgs and CouchDB via wrapper script.",
                    "Develop chaincode (`watheq_cc`) supporting `RecordDocument` and `GetDocument` with schema matching FastAPI models.",
                    "Package and deploy chaincode using latest Fabric lifecycle commands, capturing instruction log.",
                    "Expose Python (or Node) SDK helper illustrating submit/evaluate flows with sample payloads.",
                ],
                acceptance=[
                    "Network script starts cleanly and documents prerequisites (Docker, binaries, PATH).",
                    "Chaincode deployed successfully, verified via peer CLI query, and versioning strategy captured.",
                    "SDK helper invoked from FastAPI or CLI returns expected txID and payload.",
                    "Runbook updated with teardown/reset instructions and troubleshooting tips.",
                ],
                deliverables=[
                    "Infrastructure scripts under `ledger/` or `infrastructure/` for Fabric lifecycle.",
                    "Chaincode source with README (function signatures, input schema).",
                    "Sample test invoking chaincode and persisting results.",
                ],
                resources=[
                    "Hyperledger Fabric samples <https://github.com/hyperledger/fabric-samples>",
                    "Fabric chaincode tutorials <https://hyperledger-fabric.readthedocs.io/>",
                    "Execution plan section 3.5",
                ],
                dependencies=["IPFS Infrastructure and Service Wrapper"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Ledger Reliability & Retry Strategy",
            labels=["size:small", "importance:bonus", "track:trust-ledger"],
            milestone="M3: Trust & Integration Migration",
            assignee="yamam1d",
            project_fields={
                "Status": "Backlog",
                "Priority": "Medium",
                "Track": "Trust-Ledger",
                "Size": "Small",
            },
            body=format_body(
                overview=[
                    "Design resilience pattern for ledger writes, ensuring document hashes eventually land on chain even under transient outages.",
                ],
                scope=[
                    "Define retry queue schema and persistence strategy (SQL table or Redis) with exponential backoff plan.",
                    "Implement CLI or background worker to flush retries and log outcomes.",
                    "Document operational playbook for monitoring failed ledger writes and manual remediation.",
                ],
                acceptance=[
                    "Retry workflow documented and implemented with configurable intervals.",
                    "Simulated failure scenario (Fabric offline) recovers successfully via replay tool.",
                    "Operations runbook updated with monitoring and escalation steps.",
                ],
                deliverables=[
                    "Retry strategy design doc + implementation in backend.",
                    "Replay CLI script with usage instructions.",
                    "Testing notes demonstrating failure and recovery.",
                ],
                resources=[
                    "Twelve-Factor App retry guidance <https://12factor.net/backing-services>",
                    "Python retry utilities reference <https://tenacity.readthedocs.io/en/latest/>",
                ],
                dependencies=["Hyperledger Fabric Test Network & Chaincode"],
            ),
        )
    )

    # Unassigned backlog
    issues.append(
        IssueDefinition(
            title="UX Research & Figma Wireframes",
            labels=["size:medium", "importance:required", "track:frontend-web"],
            milestone="M1: Foundation & Setup",
            assignee=None,
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "Frontend-Web",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Capture user journeys and wireframes for admin and citizen experiences based on execution plan Phase 1.",
                ],
                scope=[
                    "Facilitate mini workshop with stakeholders to confirm core flows (upload, verification review, ledger audit).",
                    "Translate flows into Figma wireframes for desktop admin and mobile submission experiences.",
                    "Document accessibility and localization considerations (RTL, Arabic text density).",
                ],
                acceptance=[
                    "Wireframes reviewed with team and linked in repository docs.",
                    "Annotated flow explaining states, error messaging, and trust cues.",
                    "Accessibility checklist addressing contrast, keyboard nav, and localization.",
                ],
                deliverables=[
                    "Figma link plus exported PDF stored in docs/design/.",
                    "Journey map or user story mapping artifact.",
                ],
                resources=[
                    "Execution plan Phase 1 requirements",
                    "Gov design accessibility primer <https://designsystem.digital.gov/accessibility/>",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Quality Assurance Playbook & Test Automation",
            labels=["size:medium", "importance:required", "track:qa-docs"],
            milestone="M4: QA, Demo, and Handover",
            assignee=None,
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "QA-Docs",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Define QA strategy, automation coverage, and manual test matrix supporting execution plan Phase 4.",
                ],
                scope=[
                    "Draft QA playbook covering functional, integration, and resilience scenarios for each subsystem.",
                    "Automate smoke tests (backend pytest, web vitest, mobile widget tests) with CI hooks.",
                    "Document bug triage workflow and severity definitions aligned with academic expectations.",
                ],
                acceptance=[
                    "QA playbook stored in docs/test_plan.md with sign-off columns.",
                    "Automation scripts runnable locally and via GitHub Actions (workflow stub acceptable).",
                    "Bug triage rubric reviewed by team leads and referenced in governance docs.",
                ],
                deliverables=[
                    "Test plan markdown with matrices and schedules.",
                    "Automation scripts or job definitions committed to repo.",
                ],
                resources=[
                    "Execution plan Phase 4 testing points",
                    "GitHub Actions starter workflows <https://github.com/actions/starter-workflows>",
                ],
                dependencies=[
                    "FastAPI Backend Architecture & Database Schema",
                    "Next.js Admin Console Migration",
                ],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Academic Documentation Pack",
            labels=["size:medium", "importance:required", "track:qa-docs"],
            milestone="M4: QA, Demo, and Handover",
            assignee=None,
            project_fields={
                "Status": "Backlog",
                "Priority": "High",
                "Track": "QA-Docs",
                "Size": "Medium",
            },
            body=format_body(
                overview=[
                    "Compile academic deliverables (reports, bilingual user guides, presentation deck) required for grading per contract and plan.",
                ],
                scope=[
                    "Draft English + Arabic user manuals covering admin and citizen journeys with annotated screenshots.",
                    "Assemble operations runbook (start/stop services, ledger recovery, dataset refresh).",
                    "Prepare final presentation deck and executive summary aligning with rubric (problem, approach, results, future work).",
                ],
                acceptance=[
                    "Docs stored under docs/ with version control and review history.",
                    "Presentation deck rehearsed with notes appended or script recorded.",
                    "Document translation reviewed by bilingual team member.",
                ],
                deliverables=[
                    "docs/user_manual.md, docs/runbook.md, presentation deck assets.",
                    "Executive summary one-pager for faculty submission.",
                ],
                resources=[
                    "Execution plan Phase 4 documentation tasks",
                    "University thesis formatting guidelines (if provided)",
                ],
                dependencies=["Quality Assurance Playbook & Test Automation"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Risk Register and Mitigation Log",
            labels=["size:small", "importance:bonus", "track:governance"],
            milestone="M1: Foundation & Setup",
            assignee=None,
            project_fields={
                "Status": "Backlog",
                "Priority": "Medium",
                "Track": "Governance",
                "Size": "Small",
            },
            body=format_body(
                overview=[
                    "Establish central risk register capturing technical, schedule, and academic compliance risks.",
                ],
                scope=[
                    "Populate initial risks from execution plan (infrastructure, data availability, integration).",
                    "Define mitigation owners, triggers, and contingency actions.",
                    "Set weekly review cadence and integrate with governance hub.",
                ],
                acceptance=[
                    "Risk log accessible in hub with owners and status fields.",
                    "Top 5 risks include mitigation notes and review dates.",
                    "Cadence documented in governance README.",
                ],
                deliverables=[
                    "Risk register (Notion/Sheet) linked in docs/governance.md.",
                    "Update process documented for future risks.",
                ],
                resources=[
                    "Execution plan Phase 0 kickoff checklist",
                    "PMI risk management basics <https://www.pmi.org/learning/library/project-risk-management-8324>",
                ],
                dependencies=["Project Governance and Branch Hygiene"],
            ),
        )
    )

    issues.append(
        IssueDefinition(
            title="Analytics & Reporting Dashboard",
            labels=["size:small", "importance:bonus", "track:frontend-web"],
            milestone="M2: Core Feature Delivery",
            assignee=None,
            project_fields={
                "Status": "Backlog",
                "Priority": "Medium",
                "Track": "Frontend-Web",
                "Size": "Small",
            },
            body=format_body(
                overview=[
                    "Prototype lightweight analytics dashboards for supervisors (verification throughput, tamper trends).",
                ],
                scope=[
                    "Design metrics to display (daily verifications, tamper rate, ledger latency).",
                    "Implement React components using charting library (e.g., Recharts) pulling from mocked data initially.",
                    "Document API requirements for real data hookup later in semester.",
                ],
                acceptance=[
                    "Dashboard renders sample metrics with responsive layout.",
                    "Data model documented for backend to populate.",
                    "Future hooks into real API captured in issue checklist.",
                ],
                deliverables=[
                    "Dashboard components and storybook or screenshot evidence.",
                    "Documentation of metrics definitions and calculation approach.",
                ],
                resources=[
                    "Execution plan bonus analytics ideas",
                    "Recharts documentation <https://recharts.org/en-US>",
                ],
                dependencies=["Next.js Admin Console Migration"],
            ),
        )
    )

    return issues


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision the Watheq GitHub project assets."
    )
    parser.add_argument(
        "--owner",
        default="AbobakrMahdii",
        help="GitHub repository owner (default: %(default)s)",
    )
    parser.add_argument(
        "--repo", default="watheq", help="Repository name (default: %(default)s)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="Personal access token (default: env GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print operations without contacting GitHub",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full GraphQL payloads"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token and not args.dry_run:
        sys.exit(
            "Error: provide --token or set GITHUB_TOKEN unless running with --dry-run"
        )

    issues = build_issue_definitions()
    provisioner = GitHubProvisioner(
        token=args.token or "DUMMY",
        owner=args.owner,
        repo=args.repo,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    provisioner.run(issues, LABEL_DEFINITIONS, MILESTONES)

    if args.dry_run:
        print(
            "\nDry-run complete. Review operations above and re-run without --dry-run when ready."
        )
    else:
        print(
            "\nProvisioning complete. Review GitHub project, milestones, and issues for accuracy."
        )


if __name__ == "__main__":
    main()
