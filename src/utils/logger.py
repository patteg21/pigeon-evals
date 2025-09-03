import logging
import sys
from colorama import Fore, Style

def setup_logger(name: str = "utils.logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    class CustomFormatter(logging.Formatter):
        def __init__(self):
            super().__init__("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        def format(self, record: logging.LogRecord) -> str:
            log_colors = {
                logging.DEBUG: Fore.BLUE,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.RED + Style.BRIGHT,
            }
            colored_tag = f"{log_colors.get(record.levelno, '')}[mcp-server]{Style.RESET_ALL}"
            fmt = (f"{colored_tag} %(asctime)s - %(name)s - %(levelname)s - "
                   f"%(funcName)s - %(message)s")
            # reuse a plain Formatter for speed; keep same datefmt
            return logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S").format(record)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(CustomFormatter())
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("logs.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "[das-builder] %(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()
