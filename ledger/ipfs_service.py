import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class IPFSService:
    """
    HTTP-based IPFS client that talks to the Kubo API (default mapped to :15001).
    This mirrors infrastructure/ipfs_service.py so imports from `ledger.ipfs_service`
    keep working inside the API codebase.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:15001/api/v0"):
        self.base_url = base_url.rstrip("/")
        # Optional connectivity check
        try:
            r = requests.post(f"{self.base_url}/version")
            r.raise_for_status()
            info = r.json()
            logger.info("Connected to IPFS. Version=%s", info.get("Version"))
        except Exception as e:
            logger.error("Failed to connect to IPFS: %s", e)
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
        """Pin a local file to IPFS and return its CID."""
        url = f"{self.base_url}/add"
        with open(file_path, "rb") as f:
            files = {"file": f}
            r = requests.post(url, files=files)
        r.raise_for_status()
        last_line = r.text.strip().splitlines()[-1]
        data = json.loads(last_line)
        cid = data["Hash"]
        logger.info("File pinned. path=%s cid=%s", file_path, cid)
        return cid

    def pin_bytes(self, data: bytes, filename: str = "data") -> str:
        """Pin raw bytes to IPFS and return CID."""
        url = f"{self.base_url}/add"
        files = {"file": (filename, data)}
        r = requests.post(url, files=files)
        r.raise_for_status()
        last_line = r.text.strip().splitlines()[-1]
        info = json.loads(last_line)
        cid = info["Hash"]
        logger.info("Bytes pinned. cid=%s", cid)
        return cid

    def get_file(self, cid: str) -> bytes:
        """Retrieve file content from IPFS by CID."""
        url = f"{self.base_url}/cat"
        r = requests.post(url, params={"arg": cid}, stream=True)
        r.raise_for_status()
        content = r.content
        logger.info("Retrieved content for cid=%s size=%s bytes", cid, len(content))
        return content
