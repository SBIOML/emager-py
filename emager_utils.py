import logging as log

SAMPLES_FIFO_NAME = "emager_samples_fifo"
LABELS_FIFO_NAME = "emager_labels_fifo"
FS_KEY = "emager_sample_rate"
BATCH_KEY = "emager_batch"
GENERATED_SAMPLES_KEY = "emager_samples_gen_n"
TRANSFORM_KEY = "emager_transform"

DATASETS_ROOT = "/home/gabrielgagne/Documents/Datasets/"


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)
