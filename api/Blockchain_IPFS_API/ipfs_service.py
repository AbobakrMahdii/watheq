import requests
from config import IPFS_API_BASE, IPFS_ADD_ENDPOINT, IPFS_CAT_ENDPOINT

def add_file_to_ipfs(file_path: str) -> dict:
    url = IPFS_API_BASE + IPFS_ADD_ENDPOINT
    with open(file_path, "rb") as f:
        r = requests.post(url, files={"file": f})
    r.raise_for_status()
    data = r.json()
    return {
        "cid": data["Hash"],
        "name": data.get("Name", ""),
        "size": int(data.get("Size", 0))
    }

def download_from_ipfs(cid: str) -> bytes:
    url = f"{IPFS_API_BASE}{IPFS_CAT_ENDPOINT}?arg={cid}"
    r = requests.post(url)
    r.raise_for_status()
    return r.content
