import uvicorn

from api.app import app

if __name__ == "__main__":
    # Default port aligned with dashboard BACKEND_BASE_URL
    uvicorn.run(app, host="0.0.0.0", port=8012)
