from fastapi import FastAPI
from api.ipfs_router import router as ipfs_router

app = FastAPI(
    title="Watheq Ledger/API",
    version="0.1.0",
)

# نضيف راوتر IPFS تحت مسار /ipfs
app.include_router(ipfs_router)
