import logging
from logging.handlers import RotatingFileHandler
import structlog
from pathlib import Path

# ==========================================
# 1. AUTOMATICALLY CREATE THE LOG DIRECTORY
# ==========================================
#ROOT_DIR = Path(__file__).resolve().parent # This finds your current file's directory and creates a "logs" folder inside it
ROOT_DIR = Path("./")
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "app.log"

# ==========================================
# 2. SETUP BACKEND ROTATING FILE HANDLER
# ==========================================
file_handler = RotatingFileHandler(
    LOG_FILE_PATH, 
    maxBytes=10 * 1024 * 1024, # 10 Megabytes per file
    backupCount=5,             # Will keep  5 files, then delete old ones 
    encoding="utf-8"
)

# ==========================================
# 3. CONFIGURE STRUCTLOG & STANDARD LOGGING
# ==========================================
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer()
    ],
    # Routes logs safely into the Python Standard Library logger
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,    
)


# ==========================================
# 4. HOOK INTO STANDARD LOGGING SYSTEM
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler] # Add console_handler here if you want dual output
)

# Export logger instance for application-wide use
logger = structlog.get_logger()