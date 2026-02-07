"""
خدمة كشف الحيوية (Liveness).
في مشروع جامعي بسيط نستخدم فحوصات خفيفة بدلاً من نماذج ثقيلة:
- فحص حجم الصورة وعدد القنوات.
- تقدير بسيط للتباين/الإضاءة لتقليل صور الشاشة أو الورق.
- يمكن استبدال هذا لاحقًا بنموذج أكثر تقدمًا (anti-spoofing).
"""

import cv2
import numpy as np
from typing import Tuple


def simple_liveness_check(image_bytes: bytes) -> Tuple[bool, str]:
    # نحول البايتات إلى مصفوفة OpenCV
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, "صورة السيلفي غير صالحة"

    h, w, c = img.shape
    if h < 100 or w < 100 or c != 3:
        return False, "جودة الصورة منخفضة أو ليست صورة ملونة"

    # حساب تباين بسيط كدلـيل على حيوية (صور شاشة/ورق غالباً منخفضة التباين)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    if contrast < 10:
        return False, "حيوية مرفوضة: تباين منخفض (يبدو كصورة مطبوعة/شاشة)"

    # يمكن إضافة فحص حركة/وميض بالإطارات لاحقاً
    return True, "حيوية مقبولة"
