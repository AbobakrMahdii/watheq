from fastapi import FastAPI
from auth_router import router as auth_router
from admin_router import router as admin_router

app = FastAPI(title="Watheeq Backend")

# auth routes
app.include_router(auth_router)
app.include_router(admin_router)