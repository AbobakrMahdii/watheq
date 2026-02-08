"""
سكريبت بذر بيانات تجريبية لجدول citizen_records.
يضيف ~20 سجل مواطن للعرض التجريبي.

Usage:
    python -m api.seed_citizens
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.database import database, get_citizen_records_collection


MOCK_CITIZENS = [
    {
        "national_id": "01-001-12345-0",
        "full_name_ar": "أحمد محمد علي",
        "full_name_en": "Ahmed Mohammed Ali",
        "date_of_birth": "1990-05-15",
        "address": "صنعاء - شارع الزبيري",
        "issue_date": "2022-01-10",
        "expiry_date": "2027-01-10",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-002-23456-0",
        "full_name_ar": "فاطمة عبدالله حسن",
        "full_name_en": "Fatima Abdullah Hassan",
        "date_of_birth": "1985-11-20",
        "address": "عدن - المعلا",
        "issue_date": "2021-06-05",
        "expiry_date": "2026-06-05",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-003-34567-0",
        "full_name_ar": "محمد صالح القاسم",
        "full_name_en": "Mohammed Saleh Al-Qasem",
        "date_of_birth": "1978-03-08",
        "address": "تعز - شارع جمال",
        "issue_date": "2023-02-14",
        "expiry_date": "2028-02-14",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-004-45678-0",
        "full_name_ar": "نورة سعيد المقبلي",
        "full_name_en": "Noura Saeed Al-Muqbeli",
        "date_of_birth": "1995-07-22",
        "address": "إب - وسط المدينة",
        "issue_date": "2022-08-30",
        "expiry_date": "2027-08-30",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-005-56789-0",
        "full_name_ar": "عبدالرحمن ناصر الدين",
        "full_name_en": "Abdulrahman Nasser Al-Din",
        "date_of_birth": "1982-09-01",
        "address": "المكلا - حضرموت",
        "issue_date": "2020-12-01",
        "expiry_date": "2025-12-01",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-006-67890-0",
        "full_name_ar": "سارة أحمد الحميدي",
        "full_name_en": "Sara Ahmed Al-Humaidi",
        "date_of_birth": "1998-01-12",
        "address": "صنعاء - حدة",
        "issue_date": "2023-04-18",
        "expiry_date": "2028-04-18",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-007-78901-0",
        "full_name_ar": "خالد محمد الشرعبي",
        "full_name_en": "Khaled Mohammed Al-Sharabi",
        "date_of_birth": "1975-06-30",
        "address": "ذمار - وسط المدينة",
        "issue_date": "2021-11-22",
        "expiry_date": "2026-11-22",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-008-89012-0",
        "full_name_ar": "هدى علي باسلامة",
        "full_name_en": "Huda Ali Basalama",
        "date_of_birth": "1992-12-05",
        "address": "عدن - كريتر",
        "issue_date": "2022-03-08",
        "expiry_date": "2027-03-08",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-009-90123-0",
        "full_name_ar": "يوسف عبدالكريم النعماني",
        "full_name_en": "Yousef Abdulkareem Al-Numani",
        "date_of_birth": "1988-04-18",
        "address": "صعدة - وسط المدينة",
        "issue_date": "2023-07-01",
        "expiry_date": "2028-07-01",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-010-01234-0",
        "full_name_ar": "مريم حسين الزبيدي",
        "full_name_en": "Mariam Hussein Al-Zubaidi",
        "date_of_birth": "2000-10-25",
        "address": "الحديدة - شارع صنعاء",
        "issue_date": "2024-01-15",
        "expiry_date": "2029-01-15",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-011-11111-0",
        "full_name_ar": "عمر سالم بامحرز",
        "full_name_en": "Omar Salem Bamahraz",
        "date_of_birth": "1980-02-14",
        "address": "سيئون - حضرموت",
        "issue_date": "2020-09-20",
        "expiry_date": "2025-09-20",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-012-22222-0",
        "full_name_ar": "آمنة محمد الكبوس",
        "full_name_en": "Amna Mohammed Al-Kabous",
        "date_of_birth": "1993-08-07",
        "address": "صنعاء - السبعين",
        "issue_date": "2023-05-12",
        "expiry_date": "2028-05-12",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-013-33333-0",
        "full_name_ar": "علي حسن الأهدل",
        "full_name_en": "Ali Hassan Al-Ahdal",
        "date_of_birth": "1970-11-03",
        "address": "زبيد - الحديدة",
        "issue_date": "2021-02-28",
        "expiry_date": "2026-02-28",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-014-44444-0",
        "full_name_ar": "عائشة صالح الواسعي",
        "full_name_en": "Aisha Saleh Al-Wasiei",
        "date_of_birth": "1996-04-19",
        "address": "تعز - التحرير",
        "issue_date": "2022-10-05",
        "expiry_date": "2027-10-05",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-015-55555-0",
        "full_name_ar": "ماجد عبدالله الحكيمي",
        "full_name_en": "Majed Abdullah Al-Hakimi",
        "date_of_birth": "1987-07-11",
        "address": "إب - النادرة",
        "issue_date": "2023-09-22",
        "expiry_date": "2028-09-22",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-016-66666-0",
        "full_name_ar": "ليلى أحمد باحارثة",
        "full_name_en": "Layla Ahmed Bahartha",
        "date_of_birth": "1991-01-30",
        "address": "المكلا - خلف",
        "issue_date": "2021-07-14",
        "expiry_date": "2026-07-14",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-017-77777-0",
        "full_name_ar": "طارق محمد الغيلي",
        "full_name_en": "Tarek Mohammed Al-Ghaili",
        "date_of_birth": "1983-05-25",
        "address": "صنعاء - الأصبحي",
        "issue_date": "2022-12-03",
        "expiry_date": "2027-12-03",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-018-88888-0",
        "full_name_ar": "رانيا خالد المتوكل",
        "full_name_en": "Rania Khaled Al-Mutawakkil",
        "date_of_birth": "1999-09-16",
        "address": "عمران - وسط المدينة",
        "issue_date": "2024-03-01",
        "expiry_date": "2029-03-01",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-019-99999-0",
        "full_name_ar": "حسام علي الجنيد",
        "full_name_en": "Hussam Ali Al-Junaid",
        "date_of_birth": "1976-12-20",
        "address": "البيضاء - رداع",
        "issue_date": "2020-04-10",
        "expiry_date": "2025-04-10",
        "gender": "ذكر",
        "nationality": "يمني",
        "document_type": "بطاقة شخصية",
    },
    {
        "national_id": "01-020-10101-0",
        "full_name_ar": "سمية عبدالوهاب الحداد",
        "full_name_en": "Sumaya Abdulwahab Al-Haddad",
        "date_of_birth": "1994-06-08",
        "address": "لحج - الحوطة",
        "issue_date": "2023-11-17",
        "expiry_date": "2028-11-17",
        "gender": "أنثى",
        "nationality": "يمنية",
        "document_type": "بطاقة شخصية",
    },
]


async def seed():
    await database.connect()
    citizens = get_citizen_records_collection()
    inserted = 0
    skipped = 0
    for citizen in MOCK_CITIZENS:
        existing = await citizens.get_by_national_id(citizen["national_id"])
        if existing:
            skipped += 1
            continue
        await citizens.create(citizen)
        inserted += 1
    print(
        f"✅ Seeded citizen_records: {inserted} inserted, {skipped} skipped (already exist)"
    )
    await database.disconnect()


if __name__ == "__main__":
    asyncio.run(seed())
