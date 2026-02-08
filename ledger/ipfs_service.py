# Re-export IPFSService from infrastructure for backward compatibility.
# The canonical implementation lives in infrastructure/ipfs_service.py.
from infrastructure.ipfs_service import IPFSService  # noqa: F401

__all__ = ["IPFSService"]
