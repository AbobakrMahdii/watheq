"""Tests for authentication router endpoints."""

import pytest
from fastapi import status


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuthRouter:
    """Test suite for authentication endpoints."""

    async def test_register_new_user(self, client):
        """Test user registration with valid data."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "New User",
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "securepassword123",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "registered successfully"

    async def test_register_duplicate_username(self, client, test_user):
        """Test registration with duplicate username fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Another User",
                "username": test_user["username"],  # Duplicate
                "email": "different@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    async def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Another User",
                "username": "differentuser",
                "email": test_user["email"],  # Duplicate
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    async def test_register_invalid_email(self, client):
        """Test registration with invalid email format fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "username": "testuser",
                "email": "not-an-email",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_login_with_username(self, client, test_user):
        """Test login with valid username and password."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": test_user["username"], "password": test_user["password"]},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] in ["user", "admin", "super_admin"]

    async def test_login_with_email(self, client, test_user):
        """Test login with email instead of username."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["email"],  # Email in username field
                "password": test_user["password"],
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data

    async def test_login_wrong_password(self, client, test_user):
        """Test login with incorrect password fails."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": test_user["username"], "password": "wrongpassword"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_login_nonexistent_user(self, client):
        """Test login with non-existent username fails."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent", "password": "password123"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_login_suspended_user(self, client, test_user, raw_db):
        """Test login with suspended user fails."""
        # Suspend the user via raw DB (avoids cross-event-loop issue)
        await raw_db(
            "UPDATE users SET is_active = FALSE WHERE id = %s",
            (test_user["id"],),
        )

        response = client.post(
            "/api/v1/auth/login",
            data={"username": test_user["username"], "password": test_user["password"]},
        )

        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]

    async def test_get_current_user(self, client, test_user, user_token):
        """Test getting current user profile with valid token."""
        response = client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {user_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert "password" not in data

    async def test_get_current_user_no_token(self, client):
        """Test getting current user without token fails."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token fails."""
        response = client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_logout(self, client, user_token):
        """Test logout endpoint."""
        response = client.post(
            "/api/v1/auth/logout", headers={"Authorization": f"Bearer {user_token}"}
        )

        # Logout is client-side token removal, just verify endpoint exists
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
