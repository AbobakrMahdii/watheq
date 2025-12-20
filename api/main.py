from fastapi import FastAPI
from api.ipfs_router import router as ipfs_router
from api.ledger_router import router as ledger_router

app = FastAPI(title="Watheq Ledger/API", version="0.1.0")

app.include_router(ipfs_router)
app.include_router(ledger_router)
