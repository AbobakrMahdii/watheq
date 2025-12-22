from fastapi import FastAPI
from auth_router import router as auth_router

app = FastAPI(title="Watheeq Backend")

# auth routes
app.include_router(auth_router)