import numpy as np
import logging

def get_logger(saveName, logName = "Matting"):
    logger = logging.getLogger(logName)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(message)s")

    # log file stream
    handler = logging.FileHandler(saveName)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
