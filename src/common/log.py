"""Project logging utility for BIG_DATA_PROJECT.

Writes concise, structured logs to console and logs/ directory without dumping raw dataset content.
"""

import contextlib
import logging
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "BIG_DATA_PROJECT", log_file: str = "pipeline.log") -> logging.Logger:
    """Get or configure a project logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextlib.contextmanager
def log_operation(logger: logging.Logger, operation: str, table_name: str | None = None):
    """Context manager for timing and logging operations."""
    target = f"[{table_name}] " if table_name else ""
    logger.info(f"{target}Starting: {operation}")
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"{target}Completed: {operation} in {duration:.2f}s")
    except Exception as exc:
        duration = time.time() - start_time
        logger.error(f"{target}Failed: {operation} after {duration:.2f}s - Error: {exc}")
        raise
