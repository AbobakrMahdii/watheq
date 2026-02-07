import json
import subprocess
from typing import Any, List, Dict, Optional

CHAIN_NAME = "watheqchain"
STREAM_NAME = "documents"


def json_to_hex(data: Dict[str, Any]) -> str:
    """
    Convert a JSON-serializable dict to UTF-8 hex string.
    نستخدم HEX لأن multichain publish يتوقع بيانات hex، ولا نريد تخزين ملفات خام على السلسلة.
    """
    as_str = json.dumps(data, ensure_ascii=False)
    return as_str.encode("utf-8").hex()


def hex_to_json(hex_str: str) -> Dict[str, Any]:
    text = bytes.fromhex(hex_str).decode("utf-8")
    return json.loads(text)


def _run_cli(args: List[str]) -> str:
    """
    Execute multichain-cli command and return stdout.
    يستخدم subprocess لاستدعاء multichain-cli المحلي (لا Docker).
    """
    cmd = ["multichain-cli", CHAIN_NAME] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "multichain-cli failed")
    return proc.stdout.strip()


def publish_to_stream(key: str, data_hex: str) -> str:
    """
    Publish hex data to stream.
    - key: عادة اسم ملف أو معرف فريد
    - data_hex: بيانات hex
    """
    return _run_cli(["publish", STREAM_NAME, key, data_hex])


def list_stream_items() -> List[Dict[str, Any]]:
    """
    Return parsed liststreamitems output with decoded JSON if possible.
    """
    raw = _run_cli(["liststreamitems", STREAM_NAME])
    try:
        items = json.loads(raw)
    except Exception:
        return []

    parsed: List[Dict[str, Any]] = []
    for item in items:
        data_hex = item.get("data", "")
        try:
            decoded = hex_to_json(data_hex)
        except Exception:
            decoded = None
        parsed.append(
            {
                "key": item.get("key"),
                "txid": item.get("txid"),
                "confirmations": item.get("confirmations"),
                "blocktime": item.get("blocktime"),
                "data_hex": data_hex,
                "data_json": decoded,
            }
        )
    return parsed


def get_item_by_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Read latest item for a given key from the stream.
    """
    raw = _run_cli(["liststreamkeyitems", STREAM_NAME, key, "false", "1"])
    try:
        items = json.loads(raw)
    except Exception:
        return None
    if not items:
        return None
    item = items[-1]
    data_hex = item.get("data", "")
    try:
        decoded = hex_to_json(data_hex)
    except Exception:
        decoded = None
    return {
        "key": item.get("key"),
        "txid": item.get("txid"),
        "confirmations": item.get("confirmations"),
        "blocktime": item.get("blocktime"),
        "data_hex": data_hex,
        "data_json": decoded,
    }
