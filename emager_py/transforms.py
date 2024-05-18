"""
Transforms wrap end-to-end processing for EMG signals and are used throughout `emager_py` to preprocess data before use.
"""

import numpy as np

import emager_py.data_processing as dp
import emager_py.quantization as dq


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
    return dq.nroot_c(data, 3.0, 10000).astype(np.uint8)

def get_transform_decimation(transform: callable):
    """
    Get the decimation factor of SigProc function `transform`.
    """
    if isinstance(transform, str):
        transform = transforms_lut[transform]
        
    return 1000 // len(transform(np.zeros((1000, 1))))

transforms_lut = {
    "default": default_processing,
    "root": root_processing,
}
