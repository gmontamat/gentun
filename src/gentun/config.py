"""
logging configurations
"""

import logging
import sys


def setup_logging():
    """Setup default stdout logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
