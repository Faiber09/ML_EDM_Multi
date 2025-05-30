import logging


def get_logger() -> logging.Logger:
    logging_str = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=logging_str,
    )

    return logging.getLogger(__name__)


logger = get_logger()
