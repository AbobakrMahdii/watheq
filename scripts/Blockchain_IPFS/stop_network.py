import os
import subprocess
from pathlib import Path

FABRIC_SAMPLES_DIR = Path.home() / "fabric-samples"
TEST_NETWORK_DIR = FABRIC_SAMPLES_DIR / "test-network"
NETWORK_SH = TEST_NETWORK_DIR / "network.sh"
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"


def main():
    print("🛑 Stopping Hyperledger Fabric test network...")

    if not TEST_NETWORK_DIR.exists() or not NETWORK_SH.exists():
        raise SystemExit("⚠️ test-network or network.sh not found – is fabric-samples installed?")

    if not Path(GIT_BASH).exists():
        raise SystemExit(f"❌ Git Bash not found at: {GIT_BASH}")

    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"

    cmd = [GIT_BASH, str(NETWORK_SH), "down"]

    print(f"📂 Working dir: {TEST_NETWORK_DIR}")
    print("🧱 Command:", " ".join(cmd))

    try:
        subprocess.check_call(cmd, cwd=str(TEST_NETWORK_DIR), env=env)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"❌ Failed to stop network (code {e.returncode})") from e

    print("✅ Fabric test network is DOWN and cleaned up.")


if __name__ == "__main__":
    main()
