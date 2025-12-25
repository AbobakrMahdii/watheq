from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.face_router import router as face_router

app = FastAPI(title="Watheq Face API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_router)