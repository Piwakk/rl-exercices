import logging


def get_logger(name="", file_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_path = file_path

    if file_path is not None:
        if not file_path.parent.exists():
            file_path.parent.mkdir()

        file_handler = logging.FileHandler(filename=file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
