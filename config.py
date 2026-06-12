"""Configuration settings for Smart Business Intelligence Platform."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_NAME = os.getenv("DB_NAME", "smart_business_db")
    
    # SQLAlchemy connection string
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # App
    APP_TITLE = os.getenv("APP_TITLE", "Smart Business Intelligence Platform")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    
    # Create directories if they don't exist
    for dir_path in [MODELS_DIR, DATA_DIR, REPORTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)