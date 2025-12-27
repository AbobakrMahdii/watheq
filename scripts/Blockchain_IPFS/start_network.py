import os
import subprocess
from pathlib import Path

# مسار fabric-samples عندك (حسب اللي شفناه من قبل)
FABRIC_SAMPLES_DIR = Path.home() / "fabric-samples"
TEST_NETWORK_DIR = FABRIC_SAMPLES_DIR / "test-network"
NETWORK_SH = TEST_NETWORK_DIR / "network.sh"

# مسار Git Bash في ويندوز (عدليّه لو Git في مكان مختلف)
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"


def main():
    print("🚀 Starting Hyperledger Fabric test network...")

    # تأكد أن المسارات موجودة
    if not FABRIC_SAMPLES_DIR.exists():
        raise SystemExit(f"❌ fabric-samples not found at: {FABRIC_SAMPLES_DIR}")

    if not TEST_NETWORK_DIR.exists():
        raise SystemExit(f"❌ test-network folder not found at: {TEST_NETWORK_DIR}")

    if not NETWORK_SH.exists():
        raise SystemExit(f"❌ network.sh not found at: {NETWORK_SH}")

    if not Path(GIT_BASH).exists():
        raise SystemExit(f"❌ Git Bash not found at: {GIT_BASH}")

    env = os.environ.copy()

    # مهم جدًا عشان ما يلعب بالمسارات (المشكلة اللي كانت تصير لك)
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"

    cmd = [
        GIT_BASH,
        str(NETWORK_SH),
        "up",
        "createChannel",
        "-c",
        "mychannel",
        "-ca",
        "-s",
        "couchdb",
    ]

    print(f"📂 Working dir: {TEST_NETWORK_DIR}")
    print("🧱 Command:", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(TEST_NETWORK_DIR), env=env)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"❌ Network failed with exit code {e.returncode}") from e

    print("✅ Fabric test network is UP and channel 'mychannel' is created.")


if __name__ == "__main__":
    main()
