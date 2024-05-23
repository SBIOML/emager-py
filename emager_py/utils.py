import logging as log

def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)

