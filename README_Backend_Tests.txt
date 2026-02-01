═══════════════════════════════════════════════════════════════
   ✅ انتهينا من التعديلات البرمجية!
═══════════════════════════════════════════════════════════════

**الآن حان وقت اختبار التغييرات الجديدة.**

═══════════════════════════════════════════════════════════════
🚀 الخطوة 1: إعادة تشغيل Backend API
═══════════════════════════════════════════════════════════════

1.  **أغلق نافذة CMD/PowerShell** التي تشغل فيها Backend API حالياً (اضغط `Ctrl+C`).

2.  **شغّل Backend API مرة أخرى** (في نافذة CMD/PowerShell جديدة):
    ```bash
    cd "c:\Users\sadeq\Desktop\المشروع_2\‏‏watheq - نسخة"
    start_backend.bat
    ```

    *   تأكد من رؤية رسالة `Uvicorn running on http://0.0.0.0:8001`.

═══════════════════════════════════════════════════════════════
🧪 الخطوة 2: اختبار Backend Endpoints الجديدة (Swagger UI)
═══════════════════════════════════════════════════════════════

1.  **افتح متصفحك** واذهب إلى Swagger UI:
    ```
    http://localhost:8001/api/v1/docs
    ```

2.  **ابحث عن Endpoints الجديدة:**
    *   **"Document Types (Public)"**:
        *   `GET /api/document-types`: جرب تنفيذ هذا الـ Endpoint. يجب أن يعيد قائمة فارغة أو قائمة بأنواع الوثائق النشطة إذا قمت بإضافتها مسبقاً. **لا يتطلب هذا الـ Endpoint توكيلاً (Token).**

    *   **"Admin - Document Types"**:
        *   **قبل استخدام هذه الـ Endpoints، يجب أن تكون قد سجلت الدخول كـ Admin أو Super Admin وحصلت على JWT Token.**
        *   **إذا لم تكن قد سجلت الدخول:**
            1.  استخدم `POST /api/v1/auth/register` لإنشاء مستخدم Admin (مثال: `admin@watheq.com`, `admin123`, `admin`).
            2.  استخدم `POST /api/v1/auth/login` بنفس البيانات للحصول على الـ `access_token`.
            3.  انقر على زر **"Authorize"** في أعلى يمين صفحة Swagger UI، وأدخل التوكن بهذا الشكل: `Bearer <YOUR_JWT_TOKEN>`.

        *   **اختبر CRUD Endpoints (بعد التوثيق):**
            *   `POST /api/admin/document-types`: لإنشاء نوع وثيقة جديد.
                *   **مثال Body:**
                    ```json
                    {
                      "name": "بطاقة هوية",
                      "is_active": true,
                      "requires_back_image": false
                    }
                    ```
            *   `GET /api/admin/document-types`: لجلب جميع أنواع الوثائق (النشطة وغير النشطة).
            *   `GET /api/admin/document-types/{doc_type_id}`: لجلب نوع وثيقة معين باستخدام ID.
            *   `PUT /api/admin/document-types/{doc_type_id}`: لتعديل نوع وثيقة.
                *   **مثال Body:**
                    ```json
                    {
                      "is_active": false
                    }
                    ```
            *   `DELETE /api/admin/document-types/{doc_type_id}`: لحذف نوع وثيقة.

═══════════════════════════════════════════════════════════════
**الرجاء إخباري بالنتائج بعد اختبار الـ Backend Endpoints.**

إذا واجهت أي مشاكل أو أخطاء، يرجى تقديم رسالة الخطأ كاملة. 
