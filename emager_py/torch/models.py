from curses import window
import numpy as np

import lightning as L
import torch
import torch.nn as nn
import brevitas.nn as qnn
from torch.functional import F

from sklearn.metrics import accuracy_score

from emager_py import data_processing as dp


class EmagerCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes, quantization=-1, window_len=1):
        """
        Create a reference EmagerCNN model.

        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
            - quantization: bit-width of weights and activations. >=32 or <0 for no quantization
        """
        super().__init__()

        input_shape = list(input_shape)

        output_sizes = [32, 32, 32, 64, 64]

        layers = []

        if quantization < 0 or quantization >= 32:
            layers.append(nn.Conv2d(window_len, output_sizes[0], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[0]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[1]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[1], output_sizes[2], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[2]))
            layers.append(nn.ReLU())
            # layers.append(nn.MaxPool2d(1, 2))

            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    output_sizes[2] * np.prod(input_shape),
                    output_sizes[3],
                )
            )
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

            layers.append(nn.Linear(output_sizes[3], output_sizes[4]))
            layers.append(nn.BatchNorm1d(output_sizes[4]))
            layers.append(nn.ReLU())

            self.classifier = nn.Linear(output_sizes[4], num_classes)
        else:
            # FINN 0.10: QuantConv2d MUST have bias=False !!
            layers.append(qnn.QuantIdentity(bit_width=8, return_quant_tensor=True))
            layers.append(
                qnn.QuantConv2d(
                    window_len,
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
                    output_sizes[2] * np.prod(input_shape),
                    output_sizes[3],
                    bias=True,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.Dropout())

            # Tried almost everything and nothing works to force 8 bits :)
            layers.append(
                qnn.QuantLinear(
                    output_sizes[3],
                    output_sizes[4],
                    bias=True,
                    weight_bit_width=quantization,
                )
            )
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(qnn.QuantReLU(bit_width=quantization))

            # layers.append(
            #     qnn.QuantIdentity(
            #         bit_width=8,
            #         min_val=0,
            #         max_val=255,
            #     )
            # )

            self.classifier = qnn.QuantLinear(
                output_sizes[4],
                num_classes,
                bias=True,
                weight_bit_width=8,
            )

        self.fe = nn.Sequential(*layers)

    def forward(self, x):
        out = self.fe(x)
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
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
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

        output_sizes = [32, 32, 32, 32, 32]

        layers = []

        if quantization < 0 or quantization >= 32:
            layers.append(nn.Conv2d(1, output_sizes[0], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[0]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[1]))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(output_sizes[1], output_sizes[2], 3, padding=1))
            layers.append(nn.BatchNorm2d(output_sizes[2]))
            layers.append(nn.ReLU())

            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    output_sizes[2] * np.prod(input_shape),
                    output_sizes[3],
                )
            )
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

            layers.append(nn.Linear(output_sizes[3], output_sizes[4]))

        else:
            layers.append(qnn.QuantIdentity(bit_width=8, return_quant_tensor=True))
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
                    output_sizes[2] * np.prod(input_shape),
                    output_sizes[3],
                    bias=False,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantReLU(bit_width=quantization))
            layers.append(nn.BatchNorm1d(output_sizes[3]))
            layers.append(nn.Dropout())

            # Tried almost everything and nothing works to force 8 bits :)
            layers.append(
                qnn.QuantLinear(
                    output_sizes[3],
                    output_sizes[4],
                    bias=False,
                    weight_bit_width=quantization,
                )
            )
            layers.append(qnn.QuantIdentity(bit_width=8))

        self.fe = nn.Sequential(*layers)

    def forward(self, x):
        out = self.fe(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = F.triplet_margin_loss(anchor, positive, negative, margin=0.2)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = F.triplet_margin_loss(anchor, positive, negative, margin=0.2)
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
