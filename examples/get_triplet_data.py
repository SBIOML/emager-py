import emager_py.torch.datasets as etd


DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
train_dl, val_dl, test_dl = etd.get_triplet_dataloaders(
    DATASET_PATH,
    0,
    2,
    9,
)
print(len(train_dl.dataset), len(train_dl.dataset[0]))
