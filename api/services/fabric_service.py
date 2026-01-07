from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from typing import Any, List, Optional, Tuple, Union


def _bash(cmd: str) -> str:
    """
    Run a bash command and return stdout (or raise with stderr).

    - On Windows: prefers WSL (`wsl.exe -e bash -lc ...`) unless `FABRIC_BASH` is set.
    - On Linux/macOS: runs `bash -lc ...` (or `FABRIC_BASH` if set).
    """
    bash_override = os.environ.get("FABRIC_BASH")
    is_windows = platform.system().lower() == "windows"

    if bash_override:
        argv = [bash_override, "-lc", cmd]
    elif is_windows:
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl:
            raise RuntimeError(
                "WSL is not available. Install/enable WSL and a Linux distribution (e.g., Ubuntu), "
                "or set FABRIC_BASH to a bash executable path (e.g., Git Bash)."
            )
        argv = [wsl, "-e", "bash", "-lc", cmd]
    else:
        argv = ["bash", "-lc", cmd]

    p = subprocess.run(argv, capture_output=True, text=True)
    if p.returncode != 0:
        stderr = (p.stderr or "").strip()
        stdout = (p.stdout or "").strip()
        msg = stderr or stdout or "unknown error"
        if is_windows and "execvpe(/bin/bash)" in msg:
            msg = (
                f"{msg}\n"
                "WSL could not find `/bin/bash` in the default distro. "
                "Make sure you have a working WSL distro installed and set as default, "
                "or set FABRIC_BASH to a valid bash executable."
            )
        raise RuntimeError(msg)
    return (p.stdout or "").strip()


def _fabric_shell_prelude() -> str:
    """
    Make sure peer binary + config are found for ANY process (uvicorn, etc).
    """
    return r"""
set -e
export FABRIC_SAMPLES=${FABRIC_SAMPLES:-$HOME/fabric-samples}
export PATH=$FABRIC_SAMPLES/bin:$PATH
export FABRIC_CFG_PATH=$FABRIC_SAMPLES/config
export TEST_NETWORK=$FABRIC_SAMPLES/test-network
cd $TEST_NETWORK
. ./scripts/envVar.sh
""".strip()


def _build_args_json(fn: str, args: List[str]) -> str:
    # peer chaincode expects {"Args":["Fn","a","b",...]}
    payload = {"Args": [fn] + args}
    return json.dumps(payload)


def _as_strings(args: List[Any]) -> List[str]:
    out: List[str] = []
    for a in args:
        if a is None:
            out.append("")
        elif isinstance(a, (dict, list)):
            out.append(json.dumps(a, ensure_ascii=False))
        else:
            out.append(str(a))
    return out


def _normalize_call(
    maybe_org: Union[int, str],
    maybe_channel: Optional[str],
    maybe_chaincode: Optional[str],
    fn: Optional[str],
    args: Optional[List[Any]],
) -> Tuple[int, str, str, str, List[str]]:
    """
    Support BOTH:
      fabric_invoke(ORG, CHANNEL, CHAINCODE, fn, args)
      fabric_invoke(fn, args)
    """
    # style A: first arg is org number
    if isinstance(maybe_org, int):
        org = maybe_org
        channel = maybe_channel or "mychannel"
        chaincode = maybe_chaincode or "watheq"
        if not fn:
            raise ValueError("missing chaincode function name")
        a = _as_strings(args or [])
        return org, channel, chaincode, fn, a

    # style B: old/simple style fabric_invoke(fn, args)
    org = 1
    channel = "mychannel"
    chaincode = "watheq"
    fn2 = str(maybe_org)
    a = _as_strings(maybe_channel or []) if isinstance(maybe_channel, list) else _as_strings(args or [])
    return org, channel, chaincode, fn2, a


def fabric_query(
    org_or_fn: Union[int, str],
    channel: Optional[str] = None,
    chaincode: Optional[str] = None,
    fn: Optional[str] = None,
    args: Optional[List[Any]] = None,
) -> str:
    org, ch, cc, f, a = _normalize_call(org_or_fn, channel, chaincode, fn, args)
    args_json = _build_args_json(f, a)

    cmd = f"""
{_fabric_shell_prelude()}
setGlobals {org}
peer chaincode query -C "{ch}" -n "{cc}" -c '{args_json}'
"""
    return _bash(cmd)


def fabric_invoke(
    org_or_fn: Union[int, str],
    channel: Optional[str] = None,
    chaincode: Optional[str] = None,
    fn: Optional[str] = None,
    args: Optional[List[Any]] = None,
) -> str:
    org, ch, cc, f, a = _normalize_call(org_or_fn, channel, chaincode, fn, args)
    args_json = _build_args_json(f, a)

    # Use BOTH peers for endorsement (Org1 + Org2) like test-network expects
    cmd = f"""
{_fabric_shell_prelude()}
setGlobals {org}
peer chaincode invoke \
  -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "$ORDERER_CA" \
  -C "{ch}" -n "{cc}" \
  --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER0_ORG1_CA" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER0_ORG2_CA" \
  -c '{args_json}' \
  --waitForEvent
"""
    return _bash(cmd)
