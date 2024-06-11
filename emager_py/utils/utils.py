import logging as log
import numpy as np


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)


def get_transform_decimation(transform):
    """
    Get the decimation factor of SigProc function `transform`.
    """
    return 1000 // len(transform(np.zeros((1000, 1))))
