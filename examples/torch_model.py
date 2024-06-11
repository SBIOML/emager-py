import emager_py.torch.datasets as etd
import emager_py.utils.utils as eutils
import emager_py.data.transforms as etrans
import emager_py.torch.utils as etu
import emager_py.data.data_processing as dp
import emager_py.torch.models as etm
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

eutils.set_logging()

USE_CNN = False
DATASETS_PATH = "C:\GIT\Datasets\EMAGER/"

if USE_CNN:
    train, test = etd.get_lnocv_dataloaders(
        DATASETS_PATH,
        "000",
        "001",
        9,
        transform=etrans.default_processing,
    )
    val = test
    model = etm.EmagerCNN((4, 16), 6, -1)
else:
    # using FC at the end kills performance??
    train, val, test = etd.get_triplet_dataloaders(
        DATASETS_PATH,
        0,
        1,
        9,
        transform=etrans.default_processing,
        absda="train",
        val_batch=100,
    )
    """model = EmagerSCNN.load_from_checkpoint(
        "lightning_logs/version_2/checkpoints/epoch=4-step=2110.ckpt",
        input_shape=(4, 16),
        quantization=-1,
    )"""
    model = etm.EmagerSCNN((4, 16), -1)
trainer = etm.L.Trainer(
    max_epochs=5,
    # accelerator="cpu",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
)
# trainer.fit(model, train, val)
if isinstance(model, etm.EmagerSCNN):
    """
    train, test = etd.get_loocv_dataloaders(
        eutils.DATASETS_ROOT + "EMAGER/",
        0,
        1,
        8,
        transform=etrans.default_processing,
        test_batch=100,
    )"""
    ret = etu.get_all_embeddings(model, test, model.device)
    cemb = dp.get_n_shot_embeddings(*ret, 6, 10)
    print(ret[0].shape, ret[1].shape)
    model.set_target_embeddings(cemb)
trainer.test(model, test)