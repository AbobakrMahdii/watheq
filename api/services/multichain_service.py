"""
MultiChain service — communicates with the multichaind Docker container
via JSON-RPC over HTTP.  No local binary dependency.

The container is started via:
    docker-compose -f infrastructure/docker-compose.multichain.yml up -d

RPC defaults match the docker-compose environment variables.
"""

import json
import os
import requests
from typing import Any, List, Dict, Optional

CHAIN_NAME = "watheqchain"
STREAM_NAME = "documents"

# RPC settings — override via env vars if needed
RPC_HOST = os.getenv("MULTICHAIN_RPC_HOST", "127.0.0.1")
RPC_PORT = os.getenv("MULTICHAIN_RPC_PORT", "4402")
RPC_USER = os.getenv("MULTICHAIN_RPC_USER", "watheqrpc")
RPC_PASS = os.getenv("MULTICHAIN_RPC_PASS", "watheqrpcpass")
RPC_URL  = f"http://{RPC_HOST}:{RPC_PORT}"


# ── Low-level RPC helper ─────────────────────────────────────────

def _rpc(method: str, params: list | None = None) -> Any:
    """Send a JSON-RPC request to multichaind and return the result."""
    payload = {
        "jsonrpc": "1.0",
        "id": "watheq",
        "method": method,
        "params": params or [],
    }
    try:
        resp = requests.post(
            RPC_URL,
            json=payload,
            auth=(RPC_USER, RPC_PASS),
            timeout=10,
        )
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to MultiChain RPC at {RPC_URL}. "
            "Make sure the container is running: "
            "docker-compose -f infrastructure/docker-compose.multichain.yml up -d"
        )
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data["error"].get("message", "MultiChain RPC error"))
    return data.get("result")


# ── Public helpers (same signatures as before) ───────────────────

def json_to_hex(data: Dict[str, Any]) -> str:
    """Convert a JSON-serializable dict/string to UTF-8 hex for MultiChain."""
    as_str = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return as_str.encode("utf-8").hex()


def hex_to_json(hex_str: str) -> Dict[str, Any]:
    text = bytes.fromhex(hex_str).decode("utf-8")
    return json.loads(text)


def publish_to_stream(key: str, data_hex: str) -> str:
    """Publish hex-encoded data to the 'documents' stream. Returns txid."""
    return _rpc("publish", [STREAM_NAME, key, data_hex])


def list_stream_items() -> List[Dict[str, Any]]:
    """Return all items in 'documents' stream with decoded JSON."""
    try:
        items = _rpc("liststreamitems", [STREAM_NAME])
    except Exception:
        return []

    parsed: List[Dict[str, Any]] = []
    for item in items:
        data_hex = item.get("data", "")
        try:
            decoded = hex_to_json(data_hex) if data_hex else None
        except Exception:
            decoded = None
        parsed.append(
            {
                "key": item.get("key") or (item.get("keys", [None])[0]),
                "txid": item.get("txid"),
                "confirmations": item.get("confirmations"),
                "blocktime": item.get("blocktime"),
                "data_hex": data_hex,
                "data_json": decoded,
            }
        )
    return parsed


def get_item_by_key(key: str) -> Optional[Dict[str, Any]]:
    """Read the latest item for a given key from the 'documents' stream."""
    try:
        items = _rpc("liststreamkeyitems", [STREAM_NAME, key, False, 1])
    except Exception:
        return None
    if not items:
        return None
    item = items[-1]
    data_hex = item.get("data", "")
    try:
        decoded = hex_to_json(data_hex) if data_hex else None
    except Exception:
        decoded = None
    return {
        "key": item.get("key") or (item.get("keys", [None])[0]),
        "txid": item.get("txid"),
        "confirmations": item.get("confirmations"),
        "blocktime": item.get("blocktime"),
        "data_hex": data_hex,
        "data_json": decoded,
    }
