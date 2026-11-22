"""Simple standard-library logging setup."""

import logging


def setup_logging(level: str) -> None:
    """Configure basic application logging at the requested level."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
