"""
Lightweight stub IPFS service to keep the API running when the real service
is not available in this environment. The production implementation should
pin files to IPFS and retrieve them; here we only provide no-op methods so
imports succeed and the API can start.
"""


class IPFSService:
    def __init__(self):
        # No setup required for the stub.
        pass

    def healthy(self) -> bool:
        # Indicates IPFS is not actually connected in this stub.
        return False

    def pin_bytes(self, data: bytes, filename: str = "file") -> str:
        # In a real implementation this should return a CID.
        raise RuntimeError("IPFS service not configured")

    def get_file(self, cid: str) -> bytes:
        raise RuntimeError("IPFS service not configured")
