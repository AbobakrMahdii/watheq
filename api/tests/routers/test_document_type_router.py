"""Tests for document type router endpoints."""

import pytest
from fastapi import status


@pytest.mark.unit
@pytest.mark.asyncio
class TestDocumentTypeRouter:
    """Test suite for document type endpoints."""

    async def test_list_active_document_types(self, client, test_document_type):
        """Test listing active document types (public endpoint)."""
        response = client.get("/api/document-types")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        # Verify structure
        for doc_type in data:
            assert "id" in doc_type
            assert "name" in doc_type
            assert "is_active" in doc_type

    async def test_list_document_types_only_active(self, client, raw_db):
        """Test that only active document types are returned."""
        # Create inactive document type via raw DB (avoids cross-event-loop issue)
        await raw_db(
            "INSERT INTO document_types (name, folder_name, is_active) "
            "VALUES (%s, %s, %s)",
            ("Inactive Document", "inactive", False),
        )

        response = client.get("/api/document-types")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify no inactive types in response
        for doc_type in data:
            assert doc_type["is_active"] is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestAdminDocumentTypeRouter:
    """Test suite for admin document type management."""

    async def test_create_document_type_as_admin(self, client, admin_token):
        """Test creating a new document type as admin."""
        response = client.post(
            "/api/admin/document-types",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "name": "Passport",
                "folder_name": "passport",
                "is_active": True,
                "requires_back_image": False,
            },
        )

        assert response.status_code in [status.HTTP_200_OK, status.HTTP_201_CREATED]

    async def test_create_document_type_as_user(self, client, user_token):
        """Test that regular users cannot create document types."""
        response = client.post(
            "/api/admin/document-types",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"name": "Test Document", "folder_name": "test"},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_update_document_type(self, client, admin_token, test_document_type):
        """Test updating a document type."""
        response = client.put(
            f"/api/admin/document-types/{test_document_type['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Updated National ID", "folder_name": "updated_identity"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated National ID"

    async def test_deactivate_document_type(
        self, client, admin_token, test_document_type
    ):
        """Test deactivating a document type via update (no PATCH endpoint exists)."""
        response = client.put(
            f"/api/admin/document-types/{test_document_type['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "name": test_document_type["name"],
                "folder_name": test_document_type["folder_name"],
                "is_active": False,
            },
        )

        assert response.status_code == status.HTTP_200_OK

    async def test_delete_document_type(self, client, admin_token, test_document_type):
        """Test deleting a document type."""
        response = client.delete(
            f"/api/admin/document-types/{test_document_type['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
