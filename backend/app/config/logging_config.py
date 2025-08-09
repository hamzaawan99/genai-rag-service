import os
import sys
from loguru import logger
from pathlib import Path
from typing import Optional

from config.settings import settings

# Create logs directory if it doesn't exist
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default logger
logger.remove()

# Add console logger with colorization
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.DEBUG else "INFO",
    colorize=True,
    backtrace=True,
    diagnose=settings.DEBUG,
)

# Add file logger
logger.add(
    LOG_DIR / "rag_service.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=settings.DEBUG,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)

# Configure third-party loggers
for log_name in [
    "uvicorn",
    "uvicorn.error",
    "fastapi",
    "weaviate",
    "httpx",
    "httpcore",
]:
    logger.disable(log_name)
