import numpy as np
import logging as log
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from emager_py import dataset as ed
from emager_py import emager_redis as er
from emager_py import data_processing as dp
from emager_py import utils as eutils


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


def get_triplet_dataloaders(
    dataset_path,
    subject,
    train_session,
    val_rep,
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
    train_data, val_data = ed.get_intrasession_loocv_datasets(
        dataset_path, subject, train_session, val_rep
    )
    (train_data, train_labels), (val_data, val_labels) = dp.prepare_loocv_datasets(
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
    val_dl = DataLoader(TripletEmager(val_triplets), batch_size=val_batch, shuffle=True)
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
