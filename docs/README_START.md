# 🚀 دليل تشغيل مشروع Watheq

## المتطلبات الأساسية

قبل تشغيل المشروع، تأكد من تثبيت:

1. **Python 3.11+**
   ```bash
   python --version
   ```

2. **Node.js 18+**
   ```bash
   node --version
   ```

3. **Docker Desktop** (لـ IPFS)
   ```bash
   docker --version
   ```

4. **MySQL Server** (قاعدة البيانات)
   - يجب أن يكون MySQL قيد التشغيل
   - افتراضي: `localhost:3306`
   - Database: `watheq_db` (سيتم إنشاؤها تلقائياً)

## طريقة التشغيل السريعة

### الخطوة 0: تثبيت المتطلبات (مرة واحدة فقط)

```bash
# يكتشف تلقائياً وجود GPU ويثبت PyTorch المناسب
scripts\setup_python.bat
```

| الجهاز | ماذا يحدث |
|--------|-----------|
| مع GPU (NVIDIA) | يثبت PyTorch + CUDA (~2.5 GB) — التدريب سريع جداً |
| بدون GPU | يثبت PyTorch CPU (~200 MB) — التدريب أبطأ لكنه يعمل |

يمكنك أيضاً الإجبار:
```bash
scripts\setup_python.bat --gpu    # إجبار وضع GPU
scripts\setup_python.bat --cpu    # إجبار وضع CPU فقط
```

### الخطوة 1: تشغيل جميع الخدمات دفعة واحدة

```bash
# انقر نقراً مزدوجاً على الملف:
start_all.bat
```

سيتم فتح 3 نوافذ منفصلة:
- **IPFS Service** (Docker)
- **Backend API** (Python FastAPI)
- **Dashboard** (Next.js)

### الطريقة 2: تشغيل كل خدمة على حدة

#### 1. تشغيل IPFS
```bash
start_ipfs.bat
```
أو يدوياً:
```bash
docker-compose -f infrastructure/docker-compose.ledger.yml up
```

#### 2. تشغيل Backend API
```bash
start_backend.bat
```
أو يدوياً:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.unified.txt
python -m api.main
```

#### 3. تشغيل Dashboard
```bash
start_dashboard.bat
```
أو يدوياً:
```bash
cd dashboard
npm install
npm run dev
```

## الروابط بعد التشغيل

- **Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8001
- **API Documentation:** http://localhost:8001/api/v1/docs
- **IPFS API:** http://localhost:5001
- **IPFS Gateway:** http://localhost:8081

## إعداد قاعدة البيانات

### خيار 1: استخدام MySQL الموجود

تأكد من أن MySQL يعمل وأن لديك:
- Host: `localhost`
- Port: `3306`
- User: `root` (أو المستخدم المحدد)
- Password: (فارغ افتراضياً)

### خيار 2: تعديل الإعدادات

قم بإنشاء ملف `.env` في مجلد `api/`:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=watheq_db
SECRET_KEY=your_secret_key_here
```

## إنشاء مستخدم Admin أولي

بعد تشغيل Backend API، يمكنك إنشاء مستخدم Admin من خلال:

### الطريقة 1: استخدام API مباشرة

```bash
# تسجيل مستخدم جديد
curl -X POST http://localhost:8001/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Admin User",
    "email": "admin@watheq.com",
    "password": "admin123",
    "username": "admin"
  }'
```

ثم قم بتسجيل الدخول:
```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@watheq.com",
    "password": "admin123"
  }'
```

### الطريقة 2: استخدام Swagger UI

1. افتح http://localhost:8001/api/v1/docs
2. استخدم `/api/v1/auth/register` لإنشاء مستخدم
3. استخدم `/api/v1/auth/login` للحصول على Token
4. انقر على "Authorize" في Swagger وأدخل Token

## استكشاف الأخطاء

### المشكلة: Backend لا يبدأ

**التحقق:**
1. تأكد من أن Python مثبت
2. تأكد من تثبيت جميع المتطلبات: `pip install -r requirements.txt`
3. تأكد من أن MySQL يعمل
4. تحقق من ملف `.env` إذا كان موجوداً

**الحل:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.unified.txt
python -m api.main
```

### المشكلة: Dashboard لا يبدأ

**التحقق:**
1. تأكد من أن Node.js مثبت
2. تأكد من تثبيت dependencies: `npm install`
3. تحقق من ملف `dashboard/.env.local`

**الحل:**
```bash
cd dashboard
npm install
npm run dev
```

### المشكلة: IPFS لا يبدأ

**التحقق:**
1. تأكد من أن Docker Desktop يعمل
2. تحقق من أن Ports 5001 و 8081 غير مستخدمة

**الحل:**
```bash
docker ps
docker-compose -f infrastructure/docker-compose.ledger.yml down
docker-compose -f infrastructure/docker-compose.ledger.yml up
```

### المشكلة: Database Connection Error

**التحقق:**
1. تأكد من أن MySQL يعمل
2. تحقق من بيانات الاتصال في `.env`

**الحل:**
```bash
# اختبار الاتصال بـ MySQL
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS watheq_db;"
```

## إيقاف الخدمات

### إيقاف جميع الخدمات

1. أغلق نوافذ CMD المفتوحة
2. لإيقاف IPFS:
   ```bash
   docker-compose -f infrastructure/docker-compose.ledger.yml down
   ```

### إيقاف Backend أو Dashboard

ببساطة اضغط `Ctrl+C` في نافذة CMD الخاصة بالخدمة

## ملاحظات مهمة

1. **المنافذ المستخدمة:**
   - `3000` - Dashboard
   - `8001` - Backend API
   - `5001` - IPFS API
   - `8081` - IPFS Gateway
   - `3306` - MySQL

2. **البيئة التطويرية:**
   - جميع الخدمات تعمل في وضع Development
   - CORS مفتوح لجميع النطاقات (يجب تقييده في Production)

3. **الأمان:**
   - ⚠️ SECRET_KEY افتراضي موجود في الكود (يجب تغييره في Production)
   - ⚠️ كلمات المرور غير مشفرة في Database (يتم Hashing تلقائياً)

## الدعم

إذا واجهت أي مشاكل، راجع:
- `docs/تقرير_مراجعة_المشروع_الشامل.md` - للتقرير الكامل
- `docs/ملخص_المراجعة_السريع.md` - للملخص السريع
