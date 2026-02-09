"""Tests for admin router endpoints."""

import pytest
from fastapi import status


@pytest.mark.unit
@pytest.mark.asyncio
class TestAdminRouter:
    """Test suite for admin management endpoints."""

    async def test_list_users_as_admin(
        self, client, test_admin, admin_token, test_user
    ):
        """Test listing all users as admin."""
        response = client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        # Endpoint returns only users with role="user" (not admins)
        assert len(data) >= 1

    async def test_list_users_as_regular_user(self, client, user_token):
        """Test that regular users cannot list all users."""
        response = client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {user_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_list_users_unauthenticated(self, client):
        """Test that unauthenticated users cannot list users."""
        response = client.get("/api/v1/admin/users")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_create_user_as_admin(self, client, admin_token):
        """Test creating a new user as admin."""
        response = client.post(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "name": "Admin Created User",
                "username": "admincreated",
                "email": "admincreated@example.com",
                "password": "password123",
                "role": "user",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "User created"

    async def test_create_user_as_regular_user(self, client, user_token):
        """Test that regular users cannot create users."""
        response = client.post(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {user_token}"},
            json={
                "name": "Test User 2",
                "username": "testuser2",
                "email": "test2@example.com",
                "password": "password123",
                "role": "user",
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_edit_user_as_admin(self, client, test_user, admin_token):
        """Test editing user details as admin."""
        response = client.put(
            f"/api/v1/admin/users/{test_user['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Updated Name", "email": "updated@example.com"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "User updated"

    async def test_edit_nonexistent_user(self, client, admin_token):
        """Test editing non-existent user returns 404."""
        response = client.put(
            "/api/v1/admin/users/99999",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Updated Name"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_promote_user_to_admin(self, client, test_user, admin_token):
        """Test that only super_admin can promote users (admin gets 403)."""
        response = client.put(
            f"/api/v1/admin/users/{test_user['id']}/make-admin",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        # make-admin requires super_admin; admin role gets 403
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_suspend_user(self, client, test_user, admin_token):
        """Test suspending a user account."""
        response = client.put(
            f"/api/v1/admin/users/{test_user['id']}/suspend",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "User suspended"

    async def test_activate_user(self, client, test_user, admin_token, raw_db):
        """Test activating a suspended user account."""
        # First suspend the user via raw DB
        await raw_db(
            "UPDATE users SET is_active = FALSE WHERE id = %s",
            (test_user["id"],),
        )

        # Then activate
        response = client.put(
            f"/api/v1/admin/users/{test_user['id']}/activate",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "User activated"

    async def test_delete_user(self, client, test_user, admin_token):
        """Test soft-deleting a user account."""
        response = client.delete(
            f"/api/v1/admin/users/{test_user['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        # Soft delete should return success
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
