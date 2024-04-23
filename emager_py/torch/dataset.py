import numpy as np
import logging as log
import torch
from torch.utils.data import DataLoader, TensorDataset

from emager_py import dataset as ed
from emager_py import emager_redis as er
from emager_py import data_processing as dp
from emager_py import utils as eutils


def _get_generic_dataloaders(
    train_data, train_labels, test_data, test_labels, train_batch, test_batch, shuffle
):
    """
    Create dataloaders from numpy arrays.

    Returns a tuple of (train_dataloader, test_dataloader)
    """
    train_set = TensorDataset(
        torch.from_numpy(train_data.astype(np.float32)), torch.from_numpy(train_labels)
    )
    test_set = TensorDataset(
        torch.from_numpy(test_data.astype(np.float32)), torch.from_numpy(test_labels)
    )

    log.info(f"Train set length: {len(train_set)}, Test set length: {len(test_set)}")

    shuf_train, shuf_test = True, True
    if shuffle == "train":
        shuf_test = False
    elif shuffle == "test":
        shuf_train = False
    elif shuffle == "none":
        shuf_train, shuf_test = False, False

    return (
        DataLoader(train_set, train_batch, shuffle=shuf_train),
        DataLoader(test_set, test_batch, shuffle=shuf_test),
    )


def get_loocv_dataloaders(
    dataset_path,
    subject,
    session,
    left_out_rep,
    absda="train",
    shuffle="both",
    transform=None,
    train_batch=64,
    test_batch=256,
):
    """
    Load LOOCV datasets from disk and return DataLoader instances for training and testing.

    Returns a tuple of (train_dataloader, test_dataloader)
    """
    data, lo = ed.get_intrasession_loocv_datasets(
        dataset_path, subject, session, left_out_rep
    )
    (train_data, train_labels), (test_data, test_labels) = dp.prepare_loocv_datasets(
        data, lo, absda, transform
    )
    return _get_generic_dataloaders(
        train_data,
        train_labels,
        test_data,
        test_labels,
        train_batch,
        test_batch,
        shuffle,
    )


def get_redis_dataloaders(
    redis_host,
    absda="train",
    shuffle="both",
    split=0.8,
    transform=None,
    train_batch=64,
    test_batch=256,
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
        train_data,
        train_labels,
        test_data,
        test_labels,
        train_batch,
        test_batch,
        shuffle,
    )


if __name__ == "__main__":
    import emager_py.data_generator as edg
    import emager_py.streamers as es
    from emager_py.emager_redis import get_docker_redis_ip

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

    train_dl, test_dl = get_loocv_dataloaders(
        eutils.DATASETS_ROOT + "EMAGER/", "000", "002", 9
    )
    print(train_dl, test_dl)
