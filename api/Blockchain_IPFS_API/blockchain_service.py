import subprocess
import json
from config import MULTICHAIN_CLI, CHAIN_NAME, STREAM_NAME

def _run_cli(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout)
    return p.stdout.strip()

def publish_metadata(document_id: str, metadata: dict) -> str:
    payload = json.dumps({"json": metadata}, ensure_ascii=False)
    cmd = [
        MULTICHAIN_CLI,
        CHAIN_NAME,
        "publish",
        STREAM_NAME,
        document_id,
        payload,
        "offchain"
    ]
    out = _run_cli(cmd)
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1]

def get_by_key(document_id: str):
    out = _run_cli([
        MULTICHAIN_CLI,
        CHAIN_NAME,
        "liststreamkeyitems",
        STREAM_NAME,
        document_id
    ])
    return json.loads(out)
