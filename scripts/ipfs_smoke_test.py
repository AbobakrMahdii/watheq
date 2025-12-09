from pathlib import Path
from ledger.ipfs_service import IPFSService


def main() -> None:
    print("🚀 Running IPFS smoke test...")

    ipfs = IPFSService()

    if not ipfs.healthy():
        print("❌ IPFS daemon is not healthy on http://127.0.0.1:5001")
        return

    print("✅ IPFS daemon is healthy")

    # محتوى تجريبي بسيط (بايتات)
    test_content = b"hello from watheq IPFS smoke test\n"
    tmp_path = Path("ipfs_test.txt")
    tmp_path.write_bytes(test_content)
    print(f"📄 Created temp file: {tmp_path.resolve()}")

    # نرفع الملف لـ IPFS
    cid = ipfs.pin_file(str(tmp_path))
    print(f"📌 Pinned file to IPFS. CID = {cid}")

    # نرجع المحتوى من IPFS
    retrieved = ipfs.get_file(cid)
    print(f"📥 Retrieved content ({len(retrieved)} bytes)")

    # نتحقق إن المحتوى مطابق
    if retrieved == test_content:
        print("✅ Smoke test PASSED: content matches")
    else:
        print("❌ Smoke test FAILED: content mismatch")
        print("Original :", test_content)
        print("Retrieved:", retrieved)


if __name__ == "__main__":
    main()