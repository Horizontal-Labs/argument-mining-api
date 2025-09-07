import logging

# Try to use benchmark logging utilities if available
try:
    from benchmark.utils.logging_utils import get_logger, setup_logging
    # Use benchmark logging with API-specific name
    logger = setup_logging(log_level="DEBUG", progress_bar_compatible=True)
    logger.name = "argument-mining"  # Override the name for API context
    logger.info("Using benchmark logging utilities for API")
except ImportError:
    # Fallback to original API logging
    logger = logging.getLogger("argument-mining")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] (%(name)s) %(levelname)s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate logs via root/uvicorn handlers
    logger.propagate = False
    logger.info("Using standard API logging")


