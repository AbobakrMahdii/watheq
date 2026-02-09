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
RPC_URL = f"http://{RPC_HOST}:{RPC_PORT}"


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


# ── Extended RPC wrappers (blockchain dashboard) ─────────────────


def get_chain_info() -> Dict[str, Any]:
    """Return chain status: name, blocks, protocol, connections, etc."""
    return _rpc("getinfo")


def get_block_count() -> int:
    """Return the current block height."""
    return _rpc("getblockcount")


def get_block(height_or_hash) -> Dict[str, Any]:
    """Return block details by height (int) or hash (str). Verbose=1."""
    param = (
        int(height_or_hash) if str(height_or_hash).isdigit() else str(height_or_hash)
    )
    # getblock requires a hash; if numeric, get hash first
    if isinstance(param, int):
        block_hash = _rpc("getblockhash", [param])
    else:
        block_hash = param
    return _rpc("getblock", [block_hash, 1])


def get_raw_transaction(txid: str) -> Dict[str, Any]:
    """Return decoded transaction details (verbose=1)."""
    return _rpc("getrawtransaction", [txid, 1])


def list_streams() -> List[Dict[str, Any]]:
    """Return all streams on the chain."""
    return _rpc("liststreams")


def list_stream_keys(stream_name: str = STREAM_NAME) -> List[Dict[str, Any]]:
    """Return all keys in a given stream."""
    return _rpc("liststreamkeys", [stream_name])


def list_stream_items_by_stream(stream_name: str = STREAM_NAME) -> List[Dict[str, Any]]:
    """Return all items in a given stream, decoded."""
    try:
        items = _rpc("liststreamitems", [stream_name])
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


def list_stream_publishers(stream_name: str = STREAM_NAME) -> List[Dict[str, Any]]:
    """Return all publishers for a given stream."""
    return _rpc("liststreampublishers", [stream_name])


def get_peer_info() -> List[Dict[str, Any]]:
    """Return connected peers."""
    return _rpc("getpeerinfo")


def list_permissions(permissions: str = "*") -> List[Dict[str, Any]]:
    """Return address permissions."""
    return _rpc("listpermissions", [permissions])


def publish_raw_json(stream: str, key: str, data: Dict[str, Any]) -> str:
    """Publish arbitrary JSON data to any stream. Returns txid."""
    hex_payload = json_to_hex(data)
    return _rpc("publish", [stream, key, hex_payload])
