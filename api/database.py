from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URL = "mongodb+srv://halamoh2891:Ha0509092891@cluster0.ddlrgbc.mongodb.net/?appName=Cluster0"
DB_NAME = "wathiq_db"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

def get_user_collection():
    return db["users"]