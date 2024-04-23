import numpy as np
from scipy import signal

import emager_py.transforms as etrans
import emager_py.dataset as ed
import emager_py.quantization as dq
import emager_py.utils as utils


def extract_labels(data_array):
    """
    Given a 4D data array, extract the data and labels from the array.

    @param data_array of shape (labels, nb_exp, nb_samples, nb_channels)

    @return a tuple of (data, labels) arrays where `data` is 2D.
    data.dtype == np.int16, labels.dtype == np.uint8
    """
    labels, nb_exp, nb_sample, nb_channels = np.shape(data_array)
    X = np.zeros((labels * nb_exp * nb_sample, nb_channels), dtype=np.int16)
    y = np.zeros((labels * nb_exp * nb_sample), dtype=np.uint8)
    for label in range(labels):
        for experiment in range(nb_exp):
            X[
                nb_sample
                * (label * nb_exp + experiment) : nb_sample
                * (label * nb_exp + experiment + 1),
                :,
            ] = data_array[label, experiment, :, :]
            y[
                nb_sample
                * (label * nb_exp + experiment) : nb_sample
                * (label * nb_exp + experiment + 1)
            ] = label
    return X, y


def extract_labels_and_roll(data, roll_range, v_dim=4, h_dim=16):
    """
    From raw 4D `data`, extract labels and apply ABSDA to the array, returning a tuple of (data, labels), with shapes (2D, 1D)
    """
    emg, labels = extract_labels(data)
    emg_rolled = roll_data(emg, roll_range, v_dim, h_dim)
    nb_rolled = int(emg_rolled.shape[0] // labels.shape[0])
    labels_rolled = np.tile(labels, nb_rolled)
    return emg_rolled, labels_rolled


def filter_utility(data, fs=1000, Q=30, notch_freq=60):
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b_notch, a_notch, data, axis=0)


def preprocess_data(data_array, window_length=25, fs=1000, Q=30, notch_freq=60):
    """
    Given a 2D or 4D data array, preprocess the data by applying notch filter and DC removal.
    Processing is applied on `window_length` non-overlapping samples.

    @param data: the 4D data array to process, the data array has the format (n_gesture, n_repetition, n_samples, n_channels)
    @param window_length the length of the time window to use
    @param fs the sampling frequency of the data
    @param Q the quality factor of the notch filter
    @param notch_freq the frequency of the notch filter

    @return the processed 2D or 4D data array
    """

    if len(np.shape(data_array)) == 2:
        total_time_length, nb_channels = np.shape(data_array)
        nb_window = int(np.floor(total_time_length / window_length))
        output_data = np.zeros((nb_window, nb_channels))
        for curr_window in range(nb_window):
            start = curr_window * window_length
            end = (curr_window + 1) * window_length
            processed_data = data_array[start:end, :]
            processed_data = filter_utility(
                processed_data, fs=fs, Q=Q, notch_freq=notch_freq
            )
            processed_data = np.mean(
                np.absolute(processed_data - np.mean(processed_data, axis=0)),
                axis=0,
            )
            output_data[curr_window, :] = processed_data

    elif len(np.shape(data_array)) == 4:
        labels, nb_exp, total_time_length, nb_channels = np.shape(data_array)
        nb_window = int(np.floor(total_time_length / window_length))

        output_data = np.zeros((labels, nb_exp, nb_window, nb_channels))

        for label in range(labels):
            for experiment in range(nb_exp):
                for curr_window in range(nb_window):
                    start = curr_window * window_length
                    end = (curr_window + 1) * window_length
                    processed_data = data_array[label, experiment, start:end, :]
                    processed_data = filter_utility(
                        processed_data, fs=fs, Q=Q, notch_freq=notch_freq
                    )
                    processed_data = np.mean(
                        np.absolute(processed_data - np.mean(processed_data, axis=0)),
                        axis=0,
                    )
                    output_data[label, experiment, curr_window, :] = processed_data

    return output_data


def roll_data(data_array, rolled_range, v_dim=4, h_dim=16):
    """
    Given a 2D data array, rolls the data by the specified amount.

    @param data the data array to be rolled
    @param rolled_range the amount to roll the data by

    @return the rolled data array, with len(rolled) == (rolled_range*2+1)*len(data_array)
    """
    nb_sample, nb_channels = np.shape(data_array)
    roll_index = range(-rolled_range, rolled_range + 1)
    nb_out = len(roll_index) * nb_sample

    output_data = np.zeros((nb_out, nb_channels))
    for i, roll in enumerate(roll_index):
        tmp_data = _roll_array(data_array, roll, v_dim, h_dim)
        output_data[i * nb_sample : (i + 1) * nb_sample, :] = tmp_data

    return output_data


def _roll_array(
    data,
    roll,
    v_dim=4,
    h_dim=16,
):
    tmp_data = np.array(data)
    tmp_data = np.reshape(tmp_data, (-1, v_dim, h_dim))
    tmp_data = np.roll(tmp_data, roll, axis=2)
    tmp_data = np.reshape(tmp_data, (-1, v_dim * h_dim))
    return tmp_data


def compress_data(data, method="minmax"):
    """
    Given a data array, it will compress the data by the specified method.

    @param data the data array to be compressed
    @param method the method to use for compression, can be "minmax", "msb", or "smart"

    @return the compressed data array
    """
    if method == "minmax":
        return dq.normalize_min_max_c(data).astype(np.uint8)
    elif method == "msb":
        return dq.naive_bitshift_c(data, 8).astype(np.uint8)
    elif method == "smart":
        return dq.smart_bitshift_c(data, 8, 3).astype(np.uint8)
    elif method == "log":
        return dq.log_c(data, 20000).astype(np.uint8)
    elif method == "root":
        return dq.nroot_c(data, 3.0, 20000).astype(np.uint8)
    elif method == "none":
        return data
    else:
        raise ValueError("Invalid compression method")


def prepare_shuffled_datasets(
    data, shuffle=True, split=0.8, absda="train", transform=None
):
    """
    Prepare a typical shuffled dataset with no special LOOCV.

    Parameters:
        - `data`: data array with shape (n_gestures, n_repetitions, n_samples, n_features)
        - `split`: the ratio of training data to test data
        - `absda` : "train", "test", "both", or "none", which data to apply ABSDA to
        - `transform` : transformation to apply to the data, can be one of `emager_py.transforms` (str) or `callable`

    Returns a tuple of ((data, labels), (left_out, left_out_labels)) which can directly be used to create a TensorDataset and DataLoader, for example
    """

    data, labels = extract_labels(data)

    if transform is not None:
        if isinstance(transform, str):
            transform = etrans.transforms_lut[transform]
        data = transform(data)
        labels = labels[:: utils.get_transform_decimation(transform)]

    data = np.hstack((data, labels.reshape(-1, 1)))

    if shuffle:
        np.random.shuffle(data)  # in-place

    ds_len = len(data)
    ds_split = int(ds_len * split)

    train_ds = data[0:ds_split]
    test_ds = data[ds_split:]

    train_data, train_labels = np.hsplit(train_ds, [-1])
    test_data, test_labels = np.hsplit(test_ds, [-1])

    train_labels = np.reshape(train_labels, (-1,))
    test_labels = np.reshape(test_labels, (-1,))

    if absda == "train":
        train_data = roll_data(train_data, 2)
        nb_rolled = int(len(train_data) // len(train_labels))
        train_labels = np.tile(train_labels, nb_rolled)
    elif absda == "test":
        test_data = roll_data(test_data, 2)
        nb_rolled = int(len(test_data) // len(test_labels))
        test_labels = np.tile(test_labels, nb_rolled)
    elif absda == "both":
        train_data = roll_data(train_data, 2)
        nb_rolled = int(len(train_data) // len(train_labels))
        train_labels = np.tile(train_labels, nb_rolled)
        test_data = roll_data(test_data, 2)
        test_labels = np.tile(test_labels, nb_rolled)

    return (train_data, train_labels), (test_data, test_labels)


def prepare_loocv_datasets(
    data: np.ndarray,
    left_out: np.ndarray,
    absda="train",
    transform=None,
):
    """
    Prepare the LOOCV datasets.

    Parameters:
        - `data`: "non"-left out data
        - `left_out`: left out data array
        - `absda` : "train", "test", "both", or "none", which data to apply ABSDA to
        - `transform` : transformation to apply to the data, can be one of `emager_py.transforms` (str) or `callable`

    Returns a tuple of ((data, labels), (left_out, left_out_labels)) which can directly be used to create a TensorDataset and DataLoader, for example
    """
    if transform is not None:
        if isinstance(transform, str):
            transform = etrans.transforms_lut[transform]
        data = transform(data)
        left_out = transform(left_out)

    data_labels = None
    lo_labels = None
    if absda == "train":
        data, data_labels = extract_labels_and_roll(data, 2)
        left_out, lo_labels = extract_labels(left_out)
    elif absda == "test":
        data, data_labels = extract_labels(data)
        left_out, lo_labels = extract_labels_and_roll(left_out, 2)
    elif absda == "both":
        data, data_labels = extract_labels_and_roll(data, 2)
        left_out, lo_labels = extract_labels_and_roll(left_out, 2)
    else:
        data, data_labels = extract_labels(data)
        left_out, lo_labels = extract_labels(left_out)

    return (data, data_labels), (left_out, lo_labels)


def shuffle_dataset(data: np.ndarray, labels: np.ndarray, block_size: int):
    """
    Shuffle data and labels identically and return a tuple (shuffled_data, shuffled_labels)

    Params:
        - data : 2D array of samples
        - labels : 1D array of labels
        - block_size : shuffle data only in blocks of said size
    """
    labels = np.reshape(labels, (len(labels) // block_size, block_size, 1))
    data = np.reshape(data, (len(data) // block_size, block_size, data.shape[-1]))

    shuf = np.concatenate((data, labels), axis=2)
    np.random.shuffle(shuf)
    # (n_samples//n_blocks, n_blocks, 65)

    data = np.reshape(shuf[:, :, :-1], (-1, 64))
    labels = np.reshape(shuf[:, :, -1], (-1,))

    return data, labels


if __name__ == "__main__":
    data_array = ed.load_emager_data(
        utils.DATASETS_ROOT + "EMAGER/", "000", "002", differential=False
    )

    data = prepare_shuffled_datasets(
        data_array, split=0.8, absda="train", transform=etrans.default_processing
    )
