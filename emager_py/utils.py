import logging as log


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)


EMAGER_CHANNEL_MAP = (
    [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
    + [8, 18, 15, 19, 9, 25, 3, 30, 61, 37, 55, 43, 49, 46, 52, 38]
    + [63, 16, 14, 21, 11, 27, 5, 33, 62, 39, 57, 45, 51, 44, 50, 40]
    + [10, 22, 12, 24, 13, 26, 7, 28, 1, 31, 59, 32, 53, 34, 48, 36]
)
