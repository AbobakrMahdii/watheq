import hashlib
from typing import Union


def sha256_bytes(data: Union[bytes, bytearray]) -> str:
    """
    توليد بصمة SHA-256 لمحتوى الملف لمنع التكرار.
    لا نعتمد على اسم الملف لضمان التطابق الحقيقي للمحتوى.
    """
    return hashlib.sha256(data).hexdigest()
