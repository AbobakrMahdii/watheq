"""Tests for verification router endpoints."""

import pytest
from fastapi import status
import json


@pytest.mark.unit
@pytest.mark.asyncio
class TestVerificationRouter:
    """Test suite for verification endpoints."""

    async def test_start_verification_success(
        self, client, user_token, test_document_type, mock_image_file
    ):
        """Test starting a verification with valid data."""
        # Create second image file for person image
        mock_image_file.seek(0)
        person_file = mock_image_file

        response = client.post(
            "/api/v1/verifications/start",
            headers={"Authorization": f"Bearer {user_token}"},
            files={
                "document_image_front": ("document.jpg", mock_image_file, "image/jpeg"),
                "person_image": (" person.jpg", person_file, "image/jpeg"),
            },
            data={"document_type_id": test_document_type["id"]},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["status"] == "PENDING"

    async def test_start_verification_missing_files(
        self, client, user_token, test_document_type
    ):
        """Test starting verification without required files fails."""
        response = client.post(
            "/api/v1/verifications/start",
            headers={"Authorization": f"Bearer {user_token}"},
            data={"document_type_id": test_document_type["id"]},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_start_verification_unauthenticated(
        self, client, test_document_type, mock_image_file
    ):
        """Test that unauthenticated users cannot start verification."""
        mock_image_file.seek(0)

        response = client.post(
            "/api/v1/verifications/start",
            files={
                "document_image_front": ("document.jpg", mock_image_file, "image/jpeg"),
                "person_image": ("person.jpg", mock_image_file, "image/jpeg"),
            },
            data={"document_type_id": test_document_type["id"]},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_verification_status(
        self, client, user_token, test_user, test_document_type, raw_db
    ):
        """Test getting verification status by ID."""
        # Create a test verification via raw DB — user_id must match test_user
        verification_id = await raw_db(
            "INSERT INTO verifications (user_id, document_type_id, status, current_stage) "
            "VALUES (%s, %s, %s, %s)",
            (
                test_user["id"],
                test_document_type["id"],
                "PENDING",
                "DOCUMENT_IMAGE_QUALITY",
            ),
        )

        response = client.get(
            f"/api/v1/verifications/{verification_id}",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert (
            response.status_code == status.HTTP_200_OK
        ), f"Got {response.status_code}: {response.text}"
        data = response.json()
        assert data["id"] == verification_id
        assert data["status"] == "PENDING"

    async def test_get_nonexistent_verification(self, client, user_token):
        """Test getting non-existent verification returns 404."""
        response = client.get(
            "/api/v1/verifications/99999",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_verification_steps(
        self, client, user_token, test_user, test_document_type, raw_db
    ):
        """Test getting verification pipeline steps."""
        # Create a test verification via raw DB — user_id must match test_user
        verification_id = await raw_db(
            "INSERT INTO verifications (user_id, document_type_id, status) "
            "VALUES (%s, %s, %s)",
            (test_user["id"], test_document_type["id"], "PENDING"),
        )

        # Add a step
        await raw_db(
            "INSERT INTO verification_steps "
            "(verification_id, step_name, stage, status, result_data) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                verification_id,
                "Document Image Quality",
                "DOCUMENT_IMAGE_QUALITY",
                "SUCCESS",
                json.dumps({"brightness": 120, "blur_score": 85}),
            ),
        )

        response = client.get(
            f"/api/v1/verifications/{verification_id}/steps",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["stage"] == "DOCUMENT_IMAGE_QUALITY"

    async def test_list_my_verifications(
        self, client, user_token, test_user, test_document_type, raw_db
    ):
        """Test listing user's own verifications."""
        # Create test verifications via raw DB
        for i in range(3):
            await raw_db(
                "INSERT INTO verifications (user_id, document_type_id, status) "
                "VALUES (%s, %s, %s)",
                (
                    test_user["id"],
                    test_document_type["id"],
                    "SUCCESS" if i % 2 == 0 else "FAILED",
                ),
            )

        response = client.get(
            "/api/v1/verifications/my",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data or "verifications" in data or isinstance(data, list)

    async def test_list_verifications_with_status_filter(
        self, client, user_token, test_user, test_document_type, raw_db
    ):
        """Test filtering verifications by status."""
        # Create verifications with different statuses via raw DB
        await raw_db(
            "INSERT INTO verifications (user_id, document_type_id, status) "
            "VALUES (%s, %s, %s)",
            (test_user["id"], test_document_type["id"], "SUCCESS"),
        )

        response = client.get(
            "/api/v1/verifications/my?status=SUCCESS",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify that results are filtered (if endpoint supports filtering)
        if isinstance(data, list):
            for verification in data:
                assert verification.get("status") == "SUCCESS"
