"""logging module for pipeline"""

import logging

def init_logging(logger_name: str, log_file_path: str):
    """init logging method"""

    logger = logging.getLogger(name=logger_name)
    logger.setLevel('DEBUG')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler(log_file_path)
    stream_handler.setLevel('DEBUG')

    formatter = logging.Formatter("[%(asctime)s %(name)s %(levelname)s %(message)s]")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
