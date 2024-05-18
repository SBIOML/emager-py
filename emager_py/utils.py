import logging as log

DATASETS_ROOT = "/home/gabrielgagne/Documents/Datasets/"


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)

