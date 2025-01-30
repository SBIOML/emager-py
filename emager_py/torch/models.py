import numpy as np

import torch
import torch.nn as nn
from torch.functional import F

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import brevitas.nn as qnn

from sklearn.metrics import accuracy_score

from emager_py import data_processing as dp


class EmagerCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes, quantization=-1):
        """
        Create a reference EmagerCNN model.

        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
            - quantization: bit-width of weights and activations. >=32 or <0 for no quantization
        """
        super().__init__()

        self.input_shape = input_shape
        self.finetune = False

        output_sizes = [16, 32, 64, 32]

        layers = []

        if quantization < 0 or quantization >= 32:
            layers.append(nn.Conv2d(1, output_sizes[0], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[0]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[1]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2))
            layers.append(nn.BatchNorm2d(output_sizes[2]))
            layers.append(nn.ReLU())

            layers.append(nn.Flatten())

            layers.append(
                nn.Linear(
                    output_sizes[2] * np.prod(self.input_shape),
                    output_sizes[3],
                )
            )
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

            self.classifier = nn.Linear(output_sizes[3], num_classes)
        else:
            # FINN 0.10: QuantConv2d MUST have bias=False !!
            layers.append(qnn.QuantIdentity())
            layers.append(
                qnn.QuantConv2d(
                    1,
                    output_sizes[0],
                    3,
                    padding=1,
                    bias=False,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm2d(output_sizes[0]))

            layers.append(
                qnn.QuantConv2d(
                    output_sizes[0],
                    output_sizes[1],
                    3,
                    padding=1,
                    bias=False,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm2d(output_sizes[1]))

            layers.append(
                qnn.QuantConv2d(
                    output_sizes[1],
                    output_sizes[2],
                    3,
                    padding=1,
                    bias=False,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm2d(output_sizes[2]))

            layers.append(nn.Flatten())

            layers.append(
                qnn.QuantLinear(
                    output_sizes[2] * np.prod(self.input_shape),
                    output_sizes[3],
                    bias=True,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.Dropout())
            layers.append(
                qnn.QuantIdentity(
                    bit_width=8,
                    min_val=0,
                    max_val=255,
                )
            )

            self.classifier = qnn.QuantLinear(
                output_sizes[3],
                num_classes,
                bias=True,
                weight_bit_width=quantization,
            )

        self.fe = nn.Sequential(*layers)

    def set_finetune(self, finetune: bool):
        self.finetune = finetune
        self.fe.requires_grad_(finetune)
        self.fe.train(not finetune)

    def forward(self, x):
        out = torch.reshape(x, (-1, 1, *self.input_shape))
        out = self.fe(out)
        logits = self.classifier(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return {"acc": acc, "loss": loss, "preds": y}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class EmagerSCNN(L.LightningModule):
    def __init__(self, quantization=-1, input_shape=(4, 16)):
        """
        Create a reference Fully Convolutional Siamese Emage model.

        Parameters:
            - quantization: bit-width of weights and activations. >=32 or <0 for no quantization
        """
        super().__init__()

        # Model definition

        self.input_shape = input_shape

        self.loss = nn.TripletMarginLoss(margin=0.2)

        output_sizes = [16, 16, 16, 32, 32]

        self.bn1 = nn.BatchNorm2d(output_sizes[0])
        self.bn2 = nn.BatchNorm2d(output_sizes[1])
        self.bn3 = nn.BatchNorm2d(output_sizes[2])
        self.flat = nn.Flatten()
        self.bn4 = nn.BatchNorm1d(output_sizes[3])
        # self.bn5 = nn.BatchNorm1d(output_sizes[4])

        if quantization < 0 or quantization >= 32:
            self.inp = nn.Identity()
            self.conv1 = nn.Conv2d(1, output_sizes[0], 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(
                output_sizes[2] * np.prod(self.input_shape),
                output_sizes[3],
            )
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(
                output_sizes[3],
                output_sizes[4],
            )
            self.out = nn.Identity()
            # self.relu5 = nn.ReLU()
            # self.fc6 = nn.Linear(
            #     output_sizes[4],
            #     output_sizes[5],
            # )
        else:
            self.inp = qnn.QuantIdentity()
            self.conv1 = qnn.QuantConv2d(
                1,
                output_sizes[0],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu1 = qnn.QuantReLU(bit_width=quantization)

            self.conv2 = qnn.QuantConv2d(
                output_sizes[0],
                output_sizes[1],
                3,
                padding=1,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu2 = qnn.QuantReLU(bit_width=quantization)

            self.conv3 = qnn.QuantConv2d(
                output_sizes[1],
                output_sizes[2],
                5,
                padding=2,
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu3 = qnn.QuantReLU(bit_width=quantization)

            self.fc4 = qnn.QuantLinear(
                output_sizes[2] * np.prod(self.input_shape),
                output_sizes[3],
                bias=False,
                weight_bit_width=quantization,
            )
            self.relu4 = qnn.QuantReLU(bit_width=quantization)

            self.fc5 = qnn.QuantLinear(
                output_sizes[3],
                output_sizes[4],
                bias=False,
                weight_bit_width=quantization,
            )
            self.out = qnn.QuantIdentity(
                bit_width=8,
                min_val=0,
                max_val=255,
            )
            # self.relu5 = qnn.QuantReLU(bit_width=quantization)

            # self.fc6 = qnn.QuantLinear(
            #     output_sizes[4],
            #     output_sizes[5],
            #     bias=True,
            #     weight_bit_width=quantization,
            # )

    def forward(self, x):
        out = self.inp(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.flat(out)
        out = self.relu4(self.bn4(self.fc4(out)))
        # out = self.relu5(self.bn5(self.fc5(out)))
        out = self.fc5(out)
        out = self.out(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = self.loss(anchor, positive, negative)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = self.loss(anchor, positive, negative, margin=0.2)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_preds = np.ndarray((0,), dtype=np.uint8)

        x, y_true = batch
        embeddings = self(x).cpu().detach().numpy()

        y = dp.cosine_similarity(embeddings, self.embeddings, True)
        y_true = y_true.cpu().detach().numpy()
        acc = accuracy_score(y_true, y, normalize=True)

        self.log("test_acc", acc)
        self.test_preds = np.concatenate((self.test_preds, y))
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def set_target_embeddings(self, embeddings):
        self.embeddings = embeddings


if __name__ == "__main__":
    import emager_py.torch.datasets as etd
    import emager_py.utils as eutils
    import emager_py.transforms as etrans
    import emager_py.torch.utils as etu
    import emager_py.data_processing as dp

    # eutils.set_logging()

    USE_CNN = False
    eutils.DATASETS_ROOT = "/Users/gabrielgagne/Documents/Datasets/"

    if USE_CNN:
        train, test = etd.get_lnocv_dataloaders(
            eutils.DATASETS_ROOT + "EMAGER/",
            "000",
            "001",
            9,
            transform=etrans.default_processing,
        )
        val = test
        model = EmagerCNN((4, 16), 6, -1)
    else:
        # using FC at the end kills performance??
        train, val, test = etd.get_triplet_dataloaders(
            eutils.DATASETS_ROOT + "EMAGER/",
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
        model = EmagerSCNN((4, 16), -1)
    trainer = L.Trainer(
        max_epochs=5,
        # accelerator="cpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    # trainer.fit(model, train, val)
    if isinstance(model, EmagerSCNN):
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
