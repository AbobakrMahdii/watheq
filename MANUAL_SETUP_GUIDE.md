# Watheq — Manual Setup Guide

> Step-by-step guide to set up and run the entire Watheq platform **without** using any automated batch scripts.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [MySQL Database Setup](#2-mysql-database-setup)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Start IPFS (Docker)](#4-start-ipfs-docker)
5. [Start MultiChain (Docker)](#5-start-multichain-docker)
6. [AI Model Training](#6-ai-model-training)
7. [Start the Backend API](#7-start-the-backend-api)
8. [Start the Admin Dashboard](#8-start-the-admin-dashboard)
9. [Configure & Run the Flutter App](#9-configure--run-the-flutter-app)
10. [Seed Test Data](#10-seed-test-data)
11. [Verify Everything Works](#11-verify-everything-works)
12. [Troubleshooting](#12-troubleshooting)
13. [Stopping Services](#13-stopping-services)

---

## 1. Prerequisites

Install the following **before** starting:

| Software              | Version               | Download Link                                   |
| --------------------- | --------------------- | ----------------------------------------------- |
| **Python**            | 3.13+                 | https://www.python.org/downloads/               |
| **Node.js**           | 18+ (LTS recommended) | https://nodejs.org/                             |
| **MySQL Server**      | 8.x                   | https://dev.mysql.com/downloads/mysql/          |
| **Docker Desktop**    | Latest                | https://www.docker.com/products/docker-desktop/ |
| **Git**               | Latest                | https://git-scm.com/                            |
| **Flutter SDK**       | 3.x                   | https://docs.flutter.dev/get-started/install    |
| **NVIDIA GPU Driver** | Latest (optional)     | https://www.nvidia.com/Download/index.aspx      |

### Verify installations

Open a terminal and run:

```bash
python --version        # Should show Python 3.13.x
node --version          # Should show v18.x or higher
npm --version           # Should show 9.x or higher
mysql --version         # Should show mysql  Ver 8.x
docker --version        # Should show Docker version 2x.x
git --version           # Should show git version 2.x
flutter --version       # Should show Flutter 3.x
```

### Optional: GPU check (for faster AI training)

```bash
nvidia-smi              # Shows GPU info if NVIDIA driver is installed
```

If `nvidia-smi` works and shows a CUDA-capable GPU, you can install PyTorch with CUDA support for 10-50× faster AI training. Otherwise, CPU training still works — it's just slower.

---

## 2. MySQL Database Setup

### 2.1 Start MySQL Server

Make sure your MySQL server is running. On Windows:

- Open **Services** (`Win+R` → `services.msc`) → Find **MySQL80** → Start it
- Or from terminal: `net start MySQL80`

### 2.2 Create the database

Log into MySQL:

```bash
mysql -u root -p
```

Then create the database:

```sql
CREATE DATABASE IF NOT EXISTS watheq_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Exit MySQL:

```sql
EXIT;
```

> **Note**: The backend will automatically create all tables on first startup. You only need to create the database itself.

### 2.3 Verify MySQL credentials

The backend expects these default credentials (**no password** for root by default):

| Setting  | Default Value |
| -------- | ------------- |
| Host     | `127.0.0.1`   |
| Port     | `3306`        |
| User     | `root`        |
| Password | _(empty)_     |
| Database | `watheq_db`   |

If your MySQL has a password, you'll need to update the connection string in `api/database.py`:

```python
DATABASE_URL = "mysql+aiomysql://root:YOUR_PASSWORD@127.0.0.1:3306/watheq_db"
```

---

## 3. Python Environment Setup

### 3.1 Navigate to the project root

```bash
cd C:\Users\YOUR_USER\Desktop\watheq
```

### 3.2 Create a virtual environment

```bash
py -3.13 -m venv .venv
```

### 3.3 Activate the virtual environment

**Windows (CMD):**

```batch
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### 3.4 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.unified.txt
```

### 3.5 Install PyTorch

#### Option A: With GPU (NVIDIA CUDA — recommended if you have a GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Option B: CPU only (no GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3.6 Verify PyTorch installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output (GPU):

```
PyTorch 2.6.0+cu124
CUDA: True
```

Expected output (CPU):

```
PyTorch 2.6.0+cpu
CUDA: False
```

---

## 4. Start IPFS (Docker)

### 4.1 Make sure Docker Desktop is running

Open Docker Desktop and wait for the engine to start.

### 4.2 Start the IPFS container

From the project root:

```bash
docker-compose -p watheq-ipfs -f infrastructure/docker-compose.ipfs.yml up -d
```

### 4.3 Verify IPFS is running

Wait 10-15 seconds, then:

```bash
curl http://127.0.0.1:15001/api/v0/id
```

You should get a JSON response with the IPFS node's peer ID. If `curl` is not available:

```bash
docker logs watheq-ipfs-ipfs-node-1
```

### IPFS Ports

| Port    | Purpose                    |
| ------- | -------------------------- |
| `14001` | Swarm (peer-to-peer)       |
| `15001` | API (used by the backend)  |
| `18080` | Gateway (HTTP file access) |

---

## 5. Start MultiChain (Docker)

### 5.1 Start the MultiChain container

```bash
docker-compose -p watheq-multichain -f infrastructure/docker-compose.multichain.yml up -d --build
```

The `--build` flag builds a custom Docker image that includes:

- MultiChain binary installation
- Chain creation (`watheqchain`)
- RPC configuration
- Stream creation (`documents`)

### 5.2 Verify MultiChain is running

Wait 20-30 seconds (first run takes time), then:

```bash
curl -s -u watheqrpc:watheqrpcpass -d "{\"method\":\"getinfo\",\"params\":[],\"id\":1}" http://127.0.0.1:4402
```

You should see a JSON response with chain info. If it doesn't work yet on first run, wait and retry — the blockchain needs time to initialize.

### MultiChain Ports

| Port   | Purpose               |
| ------ | --------------------- |
| `4402` | JSON-RPC API          |
| `4403` | Peer-to-peer protocol |

### MultiChain Credentials

| Setting      | Value           |
| ------------ | --------------- |
| RPC User     | `watheqrpc`     |
| RPC Password | `watheqrpcpass` |
| Chain Name   | `watheqchain`   |
| Stream       | `documents`     |

---

## 6. AI Model Training

> **You only need to train once.** If models are already trained (`.pt` files exist in `ai/models/weights/`), you can skip this step.

### 6.1 Check if models already exist

```bash
dir ai\models\weights\*.pt
```

If `.pt` files exist, the models are already trained. Skip to Step 7.

### 6.2 Prepare reference images

Ensure your reference images exist in the correct location:

```
ai/data/refrences/identity/
├── layout_config.yaml    # Document layout definition
├── logo.png              # Reference logo image
├── seal.png              # Reference seal image
├── full_document.png     # Full document reference
└── barcode.png           # Reference barcode
```

### 6.3 Run the training pipeline

Make sure your virtual environment is activated, then:

```bash
python ai/train_ai.py --all
```

This will:

1. **Discover** document types with `layout_config.yaml` files
2. **Generate** augmented training data (genuine + forged samples)
3. **Train** EfficientNet-B0 classifiers for each document element
4. **Learn** font profiles for text regions
5. **Save** weights to `ai/models/weights/` and fonts to `ai/models/fonts/`

### Training Duration

| Hardware               | Approximate Time |
| ---------------------- | ---------------- |
| NVIDIA GPU (RTX 3050+) | 5-15 minutes     |
| CPU only               | 30-90 minutes    |

### 6.4 Verify training output

```bash
dir ai\models\weights\*.pt
dir ai\models\fonts\*.json
```

You should see files like:

```
identity_logo_main.pt
identity_seal.pt
identity_full_document.pt
identity_barcode.pt
...
identity_text_name.json
identity_text_national_id.json
```

---

## 7. Start the Backend API

### 7.1 Make sure your virtual environment is activated

```bash
.venv\Scripts\activate
```

### 7.2 Start the API server

```bash
python -u -m api.main
```

### 7.3 What happens on first startup

The backend automatically:

1. Creates the `watheq_db` database if it doesn't exist
2. Connects to MySQL
3. Creates all 10 tables:
   - `users` — User accounts
   - `document_types` — Document types
   - `audit_logs` — Audit trail
   - `verifications` — Verification records
   - `verification_steps` — Pipeline step results
   - `document_hashes` — SHA-256 + IPFS records
   - `biometric_audit_log` — Face match records
   - `citizen_records` — Reference citizen data
   - `verification_notes` — Admin notes
   - `notifications` — Alert notifications
4. Seeds a default super admin account:
   - **Email**: `admin@admin.admin`
   - **Password**: `pass1234`
   - **Role**: `super_admin`

### 7.4 Verify the backend is running

Open your browser and go to:

```
http://localhost:8012/api/v1/docs
```

You should see the **Swagger UI** with all available API endpoints.

Or test with curl:

```bash
curl http://localhost:8012/api/v1/auth/me
```

(Should return `401 Unauthorized` since you're not logged in — that means it's working!)

### Backend URL Reference

| URL                                  | Purpose                           |
| ------------------------------------ | --------------------------------- |
| `http://localhost:8012`              | API root                          |
| `http://localhost:8012/api/v1/docs`  | Swagger UI (interactive API docs) |
| `http://localhost:8012/api/v1/redoc` | ReDoc (alternative API docs)      |

---

## 8. Start the Admin Dashboard

### 8.1 Navigate to the dashboard directory

Open a **new terminal** (keep the backend running in the previous one):

```bash
cd dashboard
```

### 8.2 Install Node.js dependencies

```bash
npm install
```

### 8.3 Create environment file

Create a `.env.local` file in the `dashboard/` directory:

**Windows (CMD):**

```batch
echo BACKEND_BASE_URL=http://localhost:8012> .env.local
```

**Or manually** create `dashboard/.env.local` with:

```env
BACKEND_BASE_URL=http://localhost:8012
```

### 8.4 Start the development server

```bash
npm run dev
```

### 8.5 Access the dashboard

Open your browser and go to:

```
http://localhost:3000
```

### 8.6 Login

Use the default super admin credentials:

| Field    | Value               |
| -------- | ------------------- |
| Email    | `admin@admin.admin` |
| Password | `pass1234`          |

You should see the admin dashboard with analytics charts (empty at first).

---

## 9. Configure & Run the Flutter App

### 9.1 Navigate to the Flutter app directory

Open a **new terminal**:

```bash
cd app
```

### 9.2 Configure the API URL

Find your computer's local IP address:

```bash
ipconfig
```

Look for your **IPv4 Address** under your active adapter (e.g., `192.168.1.100`).

Then open `app/lib/core/config/app_config.dart` and update the base URL:

```dart
static const String apiBaseUrl = 'http://YOUR_IP_ADDRESS:8012';
```

> **Why not `localhost`?** The Flutter app runs on a phone/emulator which can't reach `localhost` on your PC. You need the actual network IP.

### 9.3 Install Flutter dependencies

```bash
flutter pub get
```

### 9.4 Connect a device or start an emulator

**Android Emulator:**

```bash
flutter emulators --launch <emulator_name>
```

**Or** connect a physical Android device via USB with USB debugging enabled.

**Check connected devices:**

```bash
flutter devices
```

### 9.5 Run the app

```bash
flutter run
```

### 9.6 Verify the app works

1. The app should launch to the **Splash Screen**
2. It checks for a saved token → navigates to **Login Screen**
3. Register a new user or login with existing credentials
4. Try the verification flow:
   - Tap the verification button
   - Capture a document photo
   - Take a selfie
   - Wait for the verification result

---

## 10. Seed Test Data

### 10.1 Seed citizen records

The citizen records database is used to cross-reference OCR-extracted data. To seed test data:

```bash
python -m api.seed_citizens
```

### 10.2 Seed document types

To seed default document types (if not already present):

```bash
python -m api.seed
```

---

## 11. Verify Everything Works

### Service Health Checklist

Run these checks to verify all services are operational:

| Service        | Check Command                                                                                                   | Expected             |
| -------------- | --------------------------------------------------------------------------------------------------------------- | -------------------- |
| **MySQL**      | `mysql -u root -e "SELECT 1"`                                                                                   | `1`                  |
| **IPFS**       | `curl http://127.0.0.1:15001/api/v0/id`                                                                         | JSON with PeerID     |
| **MultiChain** | `curl -s -u watheqrpc:watheqrpcpass -d "{\"method\":\"getinfo\",\"params\":[],\"id\":1}" http://127.0.0.1:4402` | JSON with chain info |
| **Backend**    | `curl http://localhost:8012/api/v1/docs`                                                                        | HTML Swagger page    |
| **Dashboard**  | Open `http://localhost:3000`                                                                                    | Login page           |
| **Flutter**    | `flutter run`                                                                                                   | App launches         |

### End-to-End Test

1. **Register** a user via the Flutter app or via API:

   ```bash
   curl -X POST http://localhost:8012/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d "{\"name\":\"Test User\",\"username\":\"testuser\",\"email\":\"test@test.com\",\"password\":\"test1234\"}"
   ```

2. **Login**:

   ```bash
   curl -X POST http://localhost:8012/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d "{\"username\":\"test@test.com\",\"password\":\"test1234\"}"
   ```

   Save the `access_token` from the response.

3. **Start a Verification** (via the Flutter app or API):

   ```bash
   curl -X POST http://localhost:8012/api/v1/verifications/start \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "document_image_front=@/path/to/document.jpg" \
     -F "person_image=@/path/to/selfie.jpg" \
     -F "document_type_id=1"
   ```

4. **Check Verification Status**:

   ```bash
   curl http://localhost:8012/api/v1/verifications/VERIFICATION_ID \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

5. **View in Dashboard** → Login at `http://localhost:3000` → Go to Verifications

---

## 12. Troubleshooting

### Python / Virtual Environment

| Problem                        | Solution                                                                   |
| ------------------------------ | -------------------------------------------------------------------------- |
| `py -3.13` not found           | Install Python 3.13 from python.org, ensure "Add to PATH" is checked       |
| `pip install` fails on PyTorch | Use the correct index URL for your setup (cu124 for GPU, cpu for CPU-only) |
| `ModuleNotFoundError`          | Make sure the virtual environment is activated (`.venv\Scripts\activate`)  |
| Import errors after install    | Run `pip install -r requirements.unified.txt` again                        |

### MySQL

| Problem                         | Solution                                                              |
| ------------------------------- | --------------------------------------------------------------------- |
| `Can't connect to MySQL server` | Ensure MySQL service is running: `net start MySQL80`                  |
| `Access denied for user 'root'` | Check your MySQL password; update `api/database.py` connection string |
| `Unknown database 'watheq_db'`  | Create it manually: `CREATE DATABASE watheq_db;`                      |

### Docker / IPFS / MultiChain

| Problem                       | Solution                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------- | -------------- |
| `docker: command not found`   | Install Docker Desktop and restart your terminal                                        |
| Docker daemon not running     | Open Docker Desktop and wait for it to fully start                                      |
| IPFS container won't start    | Remove old container: `docker rm -f ipfs-node` then retry                               |
| MultiChain RPC not responding | Wait 30 seconds after startup; check logs: `docker logs watheq-multichain-multichain-1` |
| Port conflict                 | Check if another process uses the port: `netstat -aon                                   | findstr :4402` |

### Backend API

| Problem                         | Solution                                               |
| ------------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| `Address already in use`        | Kill the existing process on port 8012: `netstat -aon  | findstr :8012`then`taskkill /PID <PID> /F` |
| Database connection error       | Verify MySQL is running and credentials are correct    |
| `Table already exists` warnings | Normal — the backend uses `CREATE TABLE IF NOT EXISTS` |

### Dashboard

| Problem             | Solution                                                                          |
| ------------------- | --------------------------------------------------------------------------------- |
| `npm install` fails | Delete `node_modules/` and `package-lock.json`, then run `npm install` again      |
| Can't login         | Check that the backend is running on port 8012                                    |
| Blank page          | Check browser console for errors; ensure `.env.local` has the correct backend URL |

### Flutter

| Problem                      | Solution                                                                   |
| ---------------------------- | -------------------------------------------------------------------------- |
| `Connection refused`         | Update `app_config.dart` with your computer's network IP, not `localhost`  |
| `No devices found`           | Connect a device or start an emulator: `flutter emulators --launch <name>` |
| Build errors                 | Run `flutter clean && flutter pub get && flutter run`                      |
| API calls fail from emulator | Use `10.0.2.2` instead of `localhost` for Android emulator                 |

---

## 13. Stopping Services

### Stop the Backend API

Press `Ctrl+C` in the backend terminal.

### Stop the Dashboard

Press `Ctrl+C` in the dashboard terminal.

### Stop IPFS

```bash
docker-compose -p watheq-ipfs -f infrastructure/docker-compose.ipfs.yml down
```

### Stop MultiChain

```bash
docker-compose -p watheq-multichain -f infrastructure/docker-compose.multichain.yml down
```

### Stop Everything

```bash
# Stop Docker containers
docker-compose -p watheq-ipfs -f infrastructure/docker-compose.ipfs.yml down
docker-compose -p watheq-multichain -f infrastructure/docker-compose.multichain.yml down

# Press Ctrl+C in the backend and dashboard terminals
```

### Stop Flutter App

Press `q` in the Flutter terminal, or `Ctrl+C`.

---

## Summary — Startup Order

Always start services in this order:

```
1. MySQL         → Must be running first (database)
2. IPFS          → Docker container (file storage)
3. MultiChain    → Docker container (blockchain)
4. Backend API   → python -u -m api.main (port 8012)
5. Dashboard     → npm run dev (port 3000)
6. Flutter App   → flutter run (connects to backend via IP)
```

---

_This guide was generated from the Watheq project source code. For the full technical documentation, see PROJECT_DOCUMENTATION.md._
