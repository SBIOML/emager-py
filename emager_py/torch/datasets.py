import numpy as np
import logging as log
import torch
from torch.utils.data import DataLoader, TensorDataset

from emager_py import dataset as ed
from emager_py import emager_redis as er
from emager_py import data_processing as dp
from emager_py import utils as eutils


def _get_generic_dataloaders(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    train_batch: int,
    test_batch: int,
    shuffle: str,
):
    """
    Create dataloaders from numpy arrays.

    Params:
        shuffle: "train" | "test" | "both" : which data should the dataloaders shuffle

    Returns a tuple of (train_dataloader, test_dataloader)
    """
    train_set = TensorDataset(
        torch.from_numpy(train_data.astype(np.float32)), torch.from_numpy(train_labels)
    )
    test_set = TensorDataset(
        torch.from_numpy(test_data.astype(np.float32)), torch.from_numpy(test_labels)
    )

    log.info(f"Train set length: {len(train_set)}, Test set length: {len(test_set)}")

    shuf_train, shuf_test = False, False
    if shuffle == "train":
        shuf_train = True
    elif shuffle == "test":
        shuf_test = True
    elif shuffle == "both":
        shuf_train, shuf_test = True, True

    return (
        DataLoader(train_set, train_batch, shuffle=shuf_train),
        DataLoader(test_set, test_batch, shuffle=shuf_test),
    )


def get_lnocv_dataloaders(
    dataset_path,
    subject,
    session: int | list,
    left_out_rep: int | list,
    absda="train",
    shuffle="train",
    transform=None,
    train_batch=64,
    test_batch=256,
    emg_shape=(4, 16),
):
    """
    Load LNOCV datasets from disk and return DataLoader instances for training and testing.

    Returns a tuple of (train_dataloader, test_dataloader)
    """
    if not isinstance(session, list):
        session = [session]
    if not isinstance(left_out_rep, list):
        left_out_rep = [left_out_rep]

    data, lo = ed.get_lnocv_datasets(dataset_path, subject, session, left_out_rep)
    (train_data, train_labels), (test_data, test_labels) = dp.prepare_lnocv_datasets(
        data, lo, absda, transform
    )
    return _get_generic_dataloaders(
        train_data.reshape((-1, 1, *emg_shape)),
        train_labels,
        test_data.reshape((-1, 1, *emg_shape)),
        test_labels,
        train_batch,
        test_batch,
        shuffle,
    )


def get_redis_dataloaders(
    redis_host,
    absda="train",
    shuffle="train",
    split=0.8,
    transform=None,
    train_batch=64,
    test_batch=256,
    emg_shape=(4, 16),
):
    """
    Create dataloaders from a Redis database by dumping them into numpy arrays and doing some preprocessing steps.

    Returns a tuple of (train_dataloader, test_dataloader)
    """

    r = er.EmagerRedis(redis_host)
    data = r.dump_labelled_to_numpy(False)

    (train_data, train_labels), (test_data, test_labels) = dp.prepare_shuffled_datasets(
        data, True, split, absda, transform
    )

    return _get_generic_dataloaders(
        train_data.reshape((-1, 1, *emg_shape)),
        train_labels,
        test_data.reshape((-1, 1, *emg_shape)),
        test_labels,
        train_batch,
        test_batch,
        shuffle,
    )


def get_triplet_dataloaders(
    dataset_path,
    subject,
    train_session: int | list,
    val_rep: int | list,
    absda="train",
    n_triplets=6000,
    transform=None,
    train_batch=64,
    val_batch=256,
    emg_shape=(4, 16),
):
    """
    Get triplet dataloaders for training, calibrating and testing.

    Returns a tuple of (train_dataloader, test_intra_dataloader, calib_inter_dataloader, test_inter_dataloader)
    """
    train_session = int(train_session)

    # intra-session train/calib/test
    train_data, test_intra_data = ed.get_lnocv_datasets(
        dataset_path, subject, train_session, val_rep
    )
    (train_data, train_labels), (test_intra_data, test_intra_labels) = (
        dp.prepare_lnocv_datasets(train_data, test_intra_data, absda, transform)
    )
    train_data = train_data.astype(np.float32)
    test_intra_data = test_intra_data.astype(np.float32)
    calib_intra_dl, test_intra_dl = _get_generic_dataloaders(
        train_data.reshape((-1, 1, *emg_shape)),
        train_labels,
        test_intra_data.reshape((-1, 1, *emg_shape)),
        test_intra_labels,
        train_batch,
        val_batch,
        "none",
    )

    # Generate triplets
    train_triplets = dp.generate_triplets(train_data, train_labels, n_triplets)
    train_triplets = [
        torch.from_numpy(t).reshape((-1, 1, *emg_shape)) for t in train_triplets
    ]
    train_dl = DataLoader(
        TensorDataset(*train_triplets), batch_size=train_batch, shuffle=True
    )

    # inter-session
    calib_inter_data, test_inter_data = ed.get_lnocv_datasets(
        dataset_path, subject, 1 if train_session == 2 else 2, val_rep
    )
    (calib_inter_data, calib_inter_labels), (test_inter_data, test_inter_labels) = (
        dp.prepare_lnocv_datasets(calib_inter_data, test_inter_data, absda, transform)
    )
    calib_inter_data = calib_inter_data.astype(np.float32)
    test_inter_data = test_inter_data.astype(np.float32)
    calib_inter_dl, test_inter_dl = _get_generic_dataloaders(
        calib_inter_data.reshape((-1, 1, *emg_shape)),
        calib_inter_labels,
        test_inter_data.reshape((-1, 1, *emg_shape)),
        test_inter_labels,
        val_batch,
        val_batch,
        "none",
    )

    return train_dl, calib_intra_dl, test_intra_dl, calib_inter_dl, test_inter_dl


if __name__ == "__main__":
    import emager_py.data_generator as edg
    import emager_py.streamers as es
    from emager_py.emager_redis import get_docker_redis_ip

    train_dl, val_dl, test_dl = get_triplet_dataloaders(
        "/Users/gabrielgagne/Documents/Datasets/EMAGER/",
        0,
        2,
        9,
    )
    print(len(train_dl.dataset), len(train_dl.dataset[0]))

    train_dl, test_dl = get_lnocv_dataloaders(
        "/home/gabrielgagne/Documents/Datasets/EMAGER/",
        0,
        [1, 2],
        [0, 3, 4],
    )
    print(len(train_dl.dataset), len(test_dl.dataset))

    _IP = get_docker_redis_ip()
    eutils.set_logging()
    dg = edg.EmagerDataGenerator(
        es.RedisStreamer(_IP, True),
        eutils.DATASETS_ROOT + "EMAGER/",
        1000000,
        10,
        True,
    )
    dg.prepare_data("000", "001")
    dg.serve_data(False)

    train_dl, test_dl = get_redis_dataloaders(_IP, "train", "train", 0.8)

    train_dl, test_dl = get_lnocv_dataloaders(
        eutils.DATASETS_ROOT + "EMAGER/", "000", "002", 9
    )

    print(train_dl, test_dl)
