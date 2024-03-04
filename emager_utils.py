import logging as log
import numpy as np

SAMPLES_FIFO_NAME = "emager_samples_fifo"
LABELS_FIFO_NAME = "emager_labels_fifo"
FS_KEY = "emager_sample_rate"
BATCH_KEY = "emager_samples_batch"
GENERATED_SAMPLES_KEY = "emager_samples_n"
TRANSFORM_KEY = "emager_transform"
BITSTREAM_KEY = "emager_bitstream"

DATASETS_ROOT = "/home/gabrielgagne/Documents/Datasets/"


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)


def get_transform_decimation(transform):
    """
    Get the decimation factor of SigProc function `transform`.
    """
    return 1000 // len(transform(np.zeros((1000, 1))))
