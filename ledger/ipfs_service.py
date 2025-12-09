# ledger/ipfs_service.py

import logging
import requests
import json
from typing import Optional

logger = logging.getLogger(__name__)


class IPFSService:
    """
    Simple HTTP-based IPFS client talking to the Kubo API at :5001.
    No dependency on ipfshttpclient (so no version mismatch issues).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:5001/api/v0"):
        self.base_url = base_url.rstrip("/")
        # مجرد فحص مبدئي (اختياري)
        try:
            r = requests.post(f"{self.base_url}/version")
            r.raise_for_status()
            info = r.json()
            logger.info(f"Connected to IPFS. Version={info.get('Version')}")
        except Exception as e:
            logger.error(f"Failed to connect to IPFS: {e}")
            # نخلي الكلاس ينشأ، لكن أول استخدام راح يبين لو فيه مشكلة
            raise

    def healthy(self) -> bool:
        """Return True if IPFS daemon responds to /version."""
        try:
            r = requests.post(f"{self.base_url}/version")
            r.raise_for_status()
            return True
        except Exception:
            return False

    def pin_file(self, file_path: str) -> str:
        """
        Pin a local file to IPFS and return its CID.
        Uses /api/v0/add.
        """
        url = f"{self.base_url}/add"
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                r = requests.post(url, files=files)
            r.raise_for_status()

            # IPFS قد يرجّع أكثر من سطر JSON، فنأخذ آخر سطر
            last_line = r.text.strip().splitlines()[-1]
            data = json.loads(last_line)
            cid = data["Hash"]
            logger.info(f"File pinned. path={file_path}, cid={cid}")
            return cid
        except Exception as e:
            logger.error(f"Failed to pin file '{file_path}': {e}")
            raise

    def pin_bytes(self, data: bytes, filename: str = "data") -> str:
        """
        Pin raw bytes to IPFS and return CID.
        """
        url = f"{self.base_url}/add"
        try:
            files = {"file": (filename, data)}
            r = requests.post(url, files=files)
            r.raise_for_status()

            last_line = r.text.strip().splitlines()[-1]
            info = json.loads(last_line)
            cid = info["Hash"]
            logger.info(f"Bytes pinned. cid={cid}")
            return cid
        except Exception as e:
            logger.error(f"Failed to pin bytes: {e}")
            raise

    def get_file(self, cid: str) -> bytes:
        """
        Retrieve file content from IPFS by CID.
        Uses /api/v0/cat.
        """
        url = f"{self.base_url}/cat"
        try:
            r = requests.post(url, params={"arg": cid}, stream=True)
            r.raise_for_status()
            content = r.content
            logger.info(f"Retrieved content for cid={cid}, size={len(content)} bytes")
            return content
        except Exception as e:
            logger.error(f"Failed to retrieve cid={cid}: {e}")
            raise
