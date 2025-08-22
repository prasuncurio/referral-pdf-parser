import logging
import os
from typing import Optional
from .config import ParserConfig


def setup_logger(config: Optional[ParserConfig] = None, level=logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger. Uses a stream handler and optionally a file handler.
    This is idempotent in the sense that repeated calls reuse the same logger name.
    """
    cfg = config or ParserConfig()
    logger = logging.getLogger("medical_pdf_parser")
    logger.setLevel(level)

    if not logger.handlers:
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)

        # Optional file handler
        if cfg.log_to_file:
            try:
                log_dir = os.path.dirname(cfg.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            except Exception:
                pass
            file_handler = logging.FileHandler(cfg.log_file_path)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

    return logger