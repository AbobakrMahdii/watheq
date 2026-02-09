# Watheq Testing Suite

## Overview

This directory contains comprehensive tests for the Watheq document verification platform, covering:
- **Backend API** (Python/FastAPI) - Unit & integration tests
- **AI/ML Models** (PyTorch) - Model validation & pipeline tests  
- **Dashboard** (Next.js/React) - Component, API route & E2E tests
- **Mobile App** (Flutter) - Widget, service & integration tests

## Quick Start

### Run All Tests
```bash
scripts\run_all_tests.bat
```

### Run Individual Test Suites

#### Backend API Tests
``` bash
scripts\run_backend_tests.bat
# Or directly with pytest:
pytest api/tests -v --cov=api
```

#### AI/ML Tests
```bash
pytest ai/tests -v --cov=ai
```

#### Dashboard Tests
```bash
cd dashboard
npm test                    # Unit tests
npm run test:e2e           # E2E tests (requires backend running)
```

#### Mobile Tests
```bash
cd app
flutter test
flutter test --coverage    # With coverage
```

## Test Structure

### Backend API Tests (`api/tests/`)
```
api/tests/
├── conftest.py                          # Pytest fixtures & config
├── test_security.py                     # JWT & password hashing
├── routers/
│   ├── test_auth_router.py             # Authentication endpoints
│   ├── test_admin_router.py            # Admin management
│   ├── test_verification_router.py     # Verification endpoints
│   ├── test_document_type_router.py    # Document types
│   └── test_admin_audit_router.py      # Audit logs
```

**Coverage**: 80+ test cases
- User registration, login, token validation
- Admin user management (CRUD, promote, suspend)
- Verification workflow (start, status, steps, filtering)
- Document type management
- Audit log queries and exports
- RBAC and permission checks

### AI/ML Tests (`ai/tests/`)
```
ai/tests/
├── test_element_classifier.py    # Model initialization, forward pass, save/load
└── test_font_analyzer.py         # Font profile learning & verification
```

**Coverage**: 25+ test cases
- Model architecture validation
- Forward/backward pass
- GPU/CPU compatibility
- Model persistence
- Font feature extraction

### Dashboard Tests (`dashboard/__tests__/` & `dashboard/e2e/`)
```
dashboard/
├── __tests__/
│   └── pages/
│       └── login.test.tsx       # Login component tests
└── e2e/
    └── auth.spec.ts             # E2E auth & navigation flows
```

**Coverage**: 15+ test cases
- Component rendering
- User interactions
- Form validation
- Authentication flow (login, logout)
- Dashboard navigation

### Mobile Tests (`app/test/` & `app/integration_test/`)
```
app/
├── test/
│   ├── widget_test.dart         # Login screen widgets
│   └── services/
│       └── auth_service_test.dart  # Auth service mocking
└── integration_test/
    └── app_test.dart            # Full app flow tests
```

**Coverage**: 15+ test cases
- Widget rendering & interaction
- Service layer with mocks
- Text input & button taps
- Integration test structure

## Test Dependencies

### Python (Backend & AI)
- pytest
- pytest-asyncio 
- pytest-cov
- pytest-mock
- httpx
- faker

### Node.js (Dashboard)
- @testing-library/react
- @testing-library/jest-dom
- @playwright/test
- jest
- jest-environment-jsdom

### Flutter (Mobile)
- flutter_test (SDK)
- mockito
- integration_test (SDK)

## Configuration Files

- `pytest.ini` - Pytest configuration
- `dashboard/jest.config.ts` - Jest configuration
- `dashboard/playwright.config.ts` - Playwright E2E config
- `app/pubspec.yaml` - Flutter test dependencies

## Coverage Targets

- **Backend API**: > 80%
- **AI Models**: > 70%
- **Dashboard**: > 75%
- **Mobile**: > 70%

## Test Markers (Pytest)

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_db` - Requires database
- `@pytest.mark.requires_docker` - Requires Docker containers

## Running Specific Test Categories

### Backend
```bash
pytest api/tests -m unit              # Only unit tests
pytest api/tests -m "not slow"        # Skip slow tests
pytest api/tests -k auth              # Only auth-related tests
```

### Dashboard
```bash
npm test -- --testPathPattern=login   # Only login tests
npm test -- --coverage                # With coverage
npm run test:e2e -- --headed         # E2E with browser UI
```

### Mobile
```bash
flutter test test/widget_test.dart    # Only widget tests
flutter test test/services/           # Only service tests
```

## CI/CD Integration

All tests can be run in CI/CD pipelines. Example GitHub Actions workflow structure:

```yaml
jobs:
  backend-tests:
    - Setup Python
    - Install dependencies
    - Run pytest
    
  dashboard-tests:
    - Setup Node.js
    - Install dependencies
    - Run Jest & Playwright
    
  mobile-tests:
    - Setup Flutter
    - Run flutter test
```

## Troubleshooting

### Backend Tests
- **Database connection errors**: Ensure MySQL is running and credentials in `.env` are correct
- **Import errors**: Activate virtual environment: `.venv\Scripts\activate`

### Dashboard Tests
- **Module not found**: Run `npm install` in dashboard directory
- **E2E test failures**: Ensure backend is running on port 8012 and dashboard on port 3200

### Mobile Tests
- **Package not found**: Run `flutter pub get`
- **Test timeout**: Increase timeout or use `--timeout=60s`

### AI Tests
- **PyTorch not found**: Run `scripts\setup_python.bat` to install PyTorch with GPU/CPU detection
- **CUDA errors**: Tests will fall back to CPU if GPU unavailable

## Writing New Tests

### Backend Test Template
```python
import pytest
from fastapi import status

@pytest.mark.unit
@pytest.mark.asyncio
async def test_my_endpoint(client, user_token):
    response = client.get(
        "/api/v1/my-endpoint",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
```

### Dashboard Test Template
``` typescript
import { render, screen } from '@testing-library/react'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

### Flutter Test Template
```dart
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('my widget test', (WidgetTester tester) async {
    await tester.pumpWidget(MyWidget());
    expect(find.text('Hello'), findsOneWidget);
  });
}
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [Flutter Testing Guide](https://docs.flutter.dev/testing)
