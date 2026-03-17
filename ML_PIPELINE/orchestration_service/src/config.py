# src/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Service info
    app_name: str = "ML Pipeline Orchestration Service"
    app_version: str = "1.0.0"
    
    # Server config
    host: str = "0.0.0.0"
    port: int = 8000
    
    # gRPC service URLs
    data_service_url: str = "localhost:50051"
    ml_service_url: str = "localhost:50052"
    metrics_service_url: str = "localhost:50053"
    
    # File upload limits
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # CORS settings
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
