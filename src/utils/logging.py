"""Minimal, consistent logging.

``get_logger`` returns a stream logger with a uniform format and guards against
duplicate handlers (which otherwise multiply log lines when called repeatedly).
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "afpfu", level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # configure once
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
