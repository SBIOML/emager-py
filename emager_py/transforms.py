"""
Transforms wrap end-to-end processing for EMG signals and are used throughout `emager_py` to preprocess data before use.
"""

import numpy as np

import emager_py.data_processing as dp
import emager_py.quantization as dq


def filter_rect_processing(data: np.ndarray) -> np.ndarray:
    """
    Apply default processing, followed by rectification
    """
    data = np.abs(dp.filter_data(data))
    return data


def filter_rect_u8_processing(data: np.ndarray) -> np.ndarray:
    """
    Apply filtering, rectification and root u8 quantization
    """
    data = np.abs(dp.filter_data(data))
    return dq.nroot_c(data, 1.7, 8).astype(np.uint8)


def default_processing(data: np.ndarray) -> np.ndarray:
    """
    Data with shape (G, R, N, C) or (N, C)
    """
    data = dp.filter_data(data)
    if len(data.shape) == 4:
        emg_mav_shape = list(data.shape)
        emg_mav_shape[-2] = emg_mav_shape[-2] // get_transform_decimation(
            dp.extract_mav
        )
        emg_mav = np.zeros(emg_mav_shape)
        for i in range(emg_mav_shape[0]):
            emg_t = data[i].reshape(-1, 64)
            emg_mav[i] = dp.extract_mav(emg_t).reshape(*emg_mav_shape[1:])
        return emg_mav
    elif len(data.shape) == 2:
        return dp.extract_mav(data)
    else:
        raise ValueError(f"Invalid data shape {data.shape}. Expected 2D or 4D array.")


def root_processing(data: np.ndarray) -> np.ndarray:
    """
    Apply default processing, followed by root quantization
    """
    data = default_processing(data)
    return dq.nroot_c(data, 1.7, 8).astype(np.uint8)


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
    "filter_rect": filter_rect_processing,
    "filter_rect_u8": filter_rect_u8_processing,
}
