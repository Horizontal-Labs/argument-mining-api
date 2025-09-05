import logging

def log() -> logging.Logger:
    log = logging.getLogger("argument-mining")

    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] (%(name)s) %(levelname)s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

    log.setLevel(logging.DEBUG)
    # Prevent duplicate logs via root/uvicorn handlers
    log.propagate = False

    return log
