import emager_py.data.data_generator as edg
import emager_py.streamers as es
from emager_py.data.emager_redis import get_docker_redis_ip
import emager_py.utils.utils as eutils
import emager_py.torch.datasets as etd


DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
train_dl, val_dl, test_dl = etd.get_triplet_dataloaders(
    DATASET_PATH,
    0,
    2,
    9,
)
print(len(train_dl.dataset), len(train_dl.dataset[0]))

train_dl, test_dl = etd.get_lnocv_dataloaders(
    DATASET_PATH,
    0,
    [1, 2],
    [0, 3, 4],
)
print(len(train_dl.dataset), len(test_dl.dataset))

_IP = get_docker_redis_ip()
eutils.set_logging()
dg = edg.EmagerDataGenerator(
    es.RedisStreamer(_IP, True),
    DATASET_PATH,
    1000000,
    10,
    True,
)
dg.prepare_data("000", "001")
dg.serve_data(False)

train_dl, test_dl = etd.get_redis_dataloaders(_IP, "train", "train", 0.8)

train_dl, test_dl = etd.get_lnocv_dataloaders(
    DATASET_PATH, "000", "002", 9
)

print(train_dl, test_dl)