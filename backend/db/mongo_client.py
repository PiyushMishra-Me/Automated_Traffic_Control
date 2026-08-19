import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from backend.config import settings

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.is_connected = False
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
            # Verify connection
            self.client.admin.command("ping")
            self.db = self.client[settings.DATABASE_NAME]
            self.is_connected = True
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self.is_connected = False
            self.client = None
            self.db = None
            logger.warning(f"MongoDB not reachable at {settings.MONGODB_URI} ({e}). Running in in-memory mode for local dev.")

    def get_database(self):
        if not self.is_connected or self.db is None:
            self._connect()
        return self.db

db_manager = MongoDBManager()
