#!/bin/bash
set -e

CHAIN_NAME="${CHAIN_NAME:-watheqchain}"
RPC_PORT="${RPC_PORT:-4402}"
RPC_USER="${RPC_USER:-watheqrpc}"
RPC_PASS="${RPC_PASS:-watheqrpcpass}"

CHAIN_DIR="/root/.multichain/${CHAIN_NAME}"

# â”€â”€ Create chain on first run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if [ ! -f "${CHAIN_DIR}/params.dat" ]; then
    echo "[multichain] Creating blockchain '${CHAIN_NAME}'..."
    multichain-util create "${CHAIN_NAME}" \
        -default-network-port=4403 \
        -default-rpc-port="${RPC_PORT}"

    # Configure RPC credentials & allow external access
    cat >> "${CHAIN_DIR}/multichain.conf" <<EOF
rpcuser=${RPC_USER}
rpcpassword=${RPC_PASS}
rpcallowip=0.0.0.0/0
rpcport=${RPC_PORT}
EOF
    echo "[multichain] Chain created. RPC on port ${RPC_PORT}."
fi

# â”€â”€ Start daemon in foreground mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[multichain] Starting multichaind (foreground)..."
multichaind "${CHAIN_NAME}" -printtoconsole \
    -rpcuser="${RPC_USER}" \
    -rpcpassword="${RPC_PASS}" \
    -rpcallowip=0.0.0.0/0 \
    -rpcport="${RPC_PORT}" &

DAEMON_PID=$!

# Wait for daemon to become responsive (up to 30 s)
echo "[multichain] Waiting for daemon..."
for i in $(seq 1 30); do
    if multichain-cli "${CHAIN_NAME}" \
        -rpcuser="${RPC_USER}" \
        -rpcpassword="${RPC_PASS}" \
        -rpcport="${RPC_PORT}" \
        getinfo >/dev/null 2>&1; then
        echo "[multichain] Daemon ready."
        break
    fi
    sleep 1
done

# â”€â”€ Ensure 'documents' stream exists & subscribed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
multichain-cli "${CHAIN_NAME}" \
    -rpcuser="${RPC_USER}" \
    -rpcpassword="${RPC_PASS}" \
    -rpcport="${RPC_PORT}" \
    create stream documents true 2>/dev/null || true

multichain-cli "${CHAIN_NAME}" \
    -rpcuser="${RPC_USER}" \
    -rpcpassword="${RPC_PASS}" \
    -rpcport="${RPC_PORT}" \
    subscribe documents 2>/dev/null || true

echo "[multichain] Ready.  chain=${CHAIN_NAME}  rpc=0.0.0.0:${RPC_PORT}"

# Keep the container alive as long as the daemon runs
wait $DAEMON_PID
