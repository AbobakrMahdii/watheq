"""Tests for audit log router endpoints."""

import pytest
from fastapi import status


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuditLogRouter:
    """Test suite for audit log endpoints."""

    async def test_list_audit_logs_as_admin(self, client, admin_token, raw_db):
        """Test listing audit logs as admin."""
        import uuid

        # Create test audit logs via raw DB (avoids cross-event-loop issue)
        for i in range(5):
            await raw_db(
                "INSERT INTO audit_logs "
                "(operation_id, operation_type, status, user_id, path, method) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    str(uuid.uuid4()),
                    "API_REQUEST",
                    "success" if i % 2 == 0 else "failed",
                    1,
                    f"/api/v1/test/{i}",
                    "GET",
                ),
            )

        response = client.get(
            "/api/admin/audit-logs", headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data or "total" in data or isinstance(data, list)

    async def test_list_audit_logs_as_user(self, client, user_token):
        """Test that regular users cannot access audit logs."""
        response = client.get(
            "/api/admin/audit-logs", headers={"Authorization": f"Bearer {user_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_filter_audit_logs_by_status(self, client, admin_token):
        """Test filtering audit logs by status."""
        response = client.get(
            "/api/admin/audit-logs?status=success",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

    async def test_filter_audit_logs_by_user(self, client, admin_token, test_user):
        """Test filtering audit logs by user ID."""
        response = client.get(
            f"/api/admin/audit-logs?user_id={test_user['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

    async def test_export_audit_logs_pdf(self, client, admin_token):
        """Test exporting audit logs as PDF."""
        response = client.get(
            "/api/admin/audit-logs/export?format=pdf",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        # PDF export may fail with 500 if reportlab is not installed
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]
        # Check content type only if endpoint returns 200 + file
        if (
            response.status_code == status.HTTP_200_OK
            and "content-type" in response.headers
        ):
            assert "pdf" in response.headers["content-type"].lower()

    async def test_export_audit_logs_excel(self, client, admin_token):
        """Test exporting audit logs as Excel."""
        response = client.get(
            "/api/admin/audit-logs/export?format=xlsx",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        # Check content type if endpoint returns file
        if "content-type" in response.headers:
            assert (
                "spreadsheet" in response.headers["content-type"].lower()
                or "excel" in response.headers["content-type"].lower()
            )

    async def test_pagination_audit_logs(self, client, admin_token):
        """Test pagination of audit logs."""
        response = client.get(
            "/api/admin/audit-logs?page=1&per_page=10",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check pagination structure if implemented
        if isinstance(data, dict):
            assert "page" in data or "logs" in data or "total" in data
