import numpy as np
import logging as log
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from emager_py.data import dataset as ed
from emager_py.data import emager_redis as er
from emager_py.data import data_processing as dp
from emager_py.utils import utils as eutils


class TripletEmager(Dataset):
    def __init__(self, triplets):
        """
        Initialize EMaGer dataset for triplet training.

        Args:
            triplets: Tuple of three numpy arrays: (anchors, positive, negative)
        """
        self.anchors = triplets[0]
        self.positives = triplets[1]
        self.negatives = triplets[2]

    def __getitem__(self, index):
        return (self.anchors[index], self.positives[index], self.negatives[index])

    def __len__(self):
        assert len(self.anchors) == len(self.positives) == len(self.negatives)
        return len(self.anchors)


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
    shuffle="train",
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
):
    """
    Get triplet dataloaders for training and validation, and a regular dataloader for testing.
    """
    train_session = int(train_session)
    test_session = 1 if train_session == 2 else 2

    # Make train and validation data
    train_data, val_data = ed.get_lnocv_datasets(
        dataset_path, subject, train_session, val_rep
    )
    # Process and split into data and labels each set
    (train_data, train_labels), (val_data, val_labels) = dp.prepare_lnocv_datasets(
        train_data, val_data, absda, transform
    )

    # Make test data
    test_data = ed.load_emager_data(dataset_path, subject, test_session)
    if transform:
        test_data = transform(test_data)
    test_data, test_labels = dp.extract_labels(test_data)

    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    train_triplets = dp.generate_triplets(train_data, train_labels, n_triplets)
    val_triplets = dp.generate_triplets(val_data, val_labels, n_triplets // 10)

    train_dl = DataLoader(
        TripletEmager(train_triplets), batch_size=train_batch, shuffle=True
    )
    val_dl = DataLoader(
        TripletEmager(val_triplets), batch_size=val_batch, shuffle=False
    )
    _, test_dl = _get_generic_dataloaders(
        np.ndarray((0, 0)),
        np.ndarray((0, 0)),
        test_data,
        test_labels,
        train_batch,
        val_batch,
        "none",
    )

    return train_dl, val_dl, test_dl