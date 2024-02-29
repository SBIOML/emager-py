import numpy as np

import data_processing as dp
import quantization as dq


def default_processing(data: np.ndarray) -> np.ndarray:
    """
    Expects data of shape (n_gestures, n_reps, n_samples, n_ch), can also accept (n_samples, 64)

    If n_samples < 25, filtering window will be made equal to n_samples.
    """
    data_len = len(data)
    if len(data.shape) == 4:
        data_len = data.shape[2]

    if data_len < 25:
        return dp.preprocess_data(data, data_len)
    else:
        return dp.preprocess_data(data)


def root_processing(data: np.ndarray) -> np.ndarray:
    """
    Apply default processing, followed by root-3 quantization
    """
    data = default_processing(data)
    return dq.nroot_c(data, 3.0, np.max(data)).astype(np.uint8)

transforms_lut = {
    "default": default_processing,
    "root": root_processing,
}
