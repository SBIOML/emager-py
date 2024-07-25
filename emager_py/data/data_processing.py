import numpy as np
from scipy import signal

import emager_py.data.dataset as ed
import emager_py.data.quantization as dq
import emager_py.data.transforms as etrans
import emager_py.utils.utils as utils


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


def prepare_lnocv_datasets(
    train_data: np.ndarray,
    test_data: np.ndarray,
    absda="train",
    transform=None,
):
    """
    Prepare the Leave-N-Out Cross Validation datasets.

    Parameters:
        - `train_data`, `test_data`: EMaGer-compatible `numpy` arrays
        - `absda` : "train", "test", "both", or "none", which data to apply ABSDA to
        - `transform` : transformation to apply to the data, can be one of `emager_py.transforms` (str) or `callable`

    Returns a tuple of ((data, labels), (left_out, left_out_labels)) which can directly be used to create a TensorDataset and DataLoader, for example
    """
    if transform is not None:
        if isinstance(transform, str):
            transform = etrans.transforms_lut[transform]
        train_data = transform(train_data)
        test_data = transform(test_data)

    data_labels = None
    lo_labels = None
    if absda == "train":
        train_data, data_labels = extract_labels_and_roll(train_data, 2)
        test_data, lo_labels = extract_labels(test_data)
    elif absda == "test":
        train_data, data_labels = extract_labels(train_data)
        test_data, lo_labels = extract_labels_and_roll(test_data, 2)
    elif absda == "both":
        train_data, data_labels = extract_labels_and_roll(train_data, 2)
        test_data, lo_labels = extract_labels_and_roll(test_data, 2)
    else:
        train_data, data_labels = extract_labels(train_data)
        test_data, lo_labels = extract_labels(test_data)

    return (train_data, data_labels), (test_data, lo_labels)


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


def generate_triplets(data: np.ndarray, labels: np.ndarray, n: int):
    """
    Generate triplets from a NC or NHW dataset and its labels.

    Params:
        - data : 2D array of samples
        - labels : 1D array of labels
        - n : number of triplets to generate per class

    Returns a tuple of 3 datasets: (anchor, positive, negative), each with shape (n, 64)
    """

    anchor_ind = np.array([])
    positive_ind = np.array([])
    negative_ind = np.array([])

    unique_len = len(labels) // len(np.unique(labels))
    if 2 * n > unique_len:
        n = unique_len // 2
    for c in np.unique(labels):
        c_ind = np.where(labels == c)[0]
        c_ind = np.random.choice(c_ind, n * 2, replace=False)
        nc_ind = np.where(labels != c)[0]
        nc_ind = np.random.choice(nc_ind, n, replace=False)
        anchor_ind = np.append(anchor_ind, c_ind[0:n])
        positive_ind = np.append(positive_ind, c_ind[n:])
        negative_ind = np.append(negative_ind, nc_ind)
    anchor_dataset = data[anchor_ind.astype(int)]
    positive_dataset = data[positive_ind.astype(int)]
    negative_dataset = data[negative_ind.astype(int)]
    return anchor_dataset, positive_dataset, negative_dataset


def cosine_similarity(
    embeddings: np.ndarray, class_embeddings: np.ndarray, closest_class=True
) -> np.ndarray:
    """
    Cosine similarity between two embeddings.

    embeddings has shape (batch_size, embedding_size)
    class_embeddings has shape = (n_class, embedding_size)

    If closest_class is True, returns a matrix of shape (batch_size,) where each element is the index of the closest class for each embedding in the batch.

    Returns a matrix of shape (batch_size, n_class) where each row is the cosine similarity of the corresponding embedding with each class embedding

    ## Example

    >>> emb = np.random.rand(10, 64)
    >>> class_emb = np.random.rand(6, 64)
    >>> closest_class = cosine_similarity(emb, class_emb, closest_class=True)
    >>> print(closest_class, closest_class.shape)
    >>> class_similarity = cosine_similarity(emb, class_emb, closest_class=False)
    >>> print(class_similarity, class_similarity.shape)
    """
    nemb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    nembc = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)

    cos_sim = np.matmul(nemb, nembc.T)

    if closest_class:
        return np.argmax(cos_sim, axis=1)

    return cos_sim


def get_mean_embeddings(embeddings: np.ndarray, labels: np.ndarray, n_classes: int):
    """
    From a batch of embeddings and corresponding labels, generate mean embeddings for each class.

    Params:
        - embeddings : 2D array of embeddings : (N, W) where W is the embedding width
        - labels : 1D array of labels : (N,)
        - n_classes : number of target classes

    Returns a 2D array of shape (n_classes, W) where each row is the mean embedding for each class.
    """
    ret = np.zeros((n_classes, *embeddings.shape[1:]))
    for unique_y in np.unique(labels):
        t = np.where(labels == unique_y)[0]
        batch_sum = np.sum(embeddings[t], axis=0)
        ret[unique_y] += batch_sum
    return ret


def get_n_shot_embeddings(
    embeddings: np.ndarray, labels: np.ndarray, n_classes: int, n_shots: int
):
    """
    From raw embeddings and corresponding labels, generate n-shot embeddings for each class.

    Under the hood, this function samples randomly `n_shots` from each class and calls `get_mean_embeddings` to generate the mean embeddings for each class.

    Params:
        - embeddings : 2D array of embeddings : (N, W) where W is the embedding width
        - labels : 1D array of labels : (N,)
        - n_classes : number of target classes
        - n_shots : number of shots per class

    Returns a 2D array of shape (n_classes, W) where each row is the n-shot embedding for each class.
    Any subsequent embedding can then be classified against this array, for example with `emager_py.data_processing.cosine_similarity`.
    """
    assert len(embeddings) == len(labels)
    labels = labels.astype(np.uint8)

    if n_shots == -1:
        return get_mean_embeddings(embeddings, labels, n_classes)

    to_sample = np.zeros((0,), dtype=np.uint8)
    for k in np.unique(labels):
        num_k = np.sum([labels == k])
        to_sample_k = np.random.choice(
            np.where(labels == k)[0],
            min(n_shots, num_k),
            replace=False,
        )
        to_sample = np.append(to_sample, to_sample_k)

    return get_mean_embeddings(embeddings[to_sample], labels[to_sample], n_classes)

def unmap(emg: np.ndarray):
    _emg = emg.copy()
    channelMap = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
                          [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
                          [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
                          [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
    _emg[:, np.array(channelMap)] = emg[:, np.arange(len(channelMap))]
    return _emg

def unroll_starting_point(packet):
    offset = np.random.randint(0, len(packet))
    rolled_packet = np.roll(packet, offset)
    return rolled_packet