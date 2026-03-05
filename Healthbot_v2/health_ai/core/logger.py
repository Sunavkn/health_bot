import logging
from health_ai.config.settings import LOG_DIR


def setup_logger(name: str, file_name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_DIR / file_name)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

ingestion_logger = setup_logger("ingestion", "ingestion.log")
error_logger = setup_logger("error", "error.log")
