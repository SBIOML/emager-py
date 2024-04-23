import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

import emager_py.torch.utils as etu
import emager_py.torch.losses as etl
import brevitas.nn as qnn


class EmagerCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape

        output_sizes = [32, 32, 32, 256]

        # 1st layer
        self.conv1 = nn.Conv2d(1, output_sizes[0], 3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_sizes[0])

        # 2nd layer
        self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(output_sizes[1])

        # 3rd conv
        self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(output_sizes[2])

        self.flat = nn.Flatten()

        # Fully connected
        self.fc4 = nn.Linear(
            output_sizes[2] * np.prod(self.input_shape),
            output_sizes[3],
        )
        self.dropout4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(output_sizes[3])

        self.fc5 = nn.Linear(output_sizes[3], num_classes)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, *self.input_shape))
        out = x
        out = self.bn1(self.relu1(self.conv1(out)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.bn4(self.relu4(self.dropout4(self.fc4(out))))
        logits = self.fc5(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = self.loss(y, y_true)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class QEmagerCNN(L.LightningModule):
    def __init__(self, bit_width, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape
        self.loss = nn.CrossEntropyLoss()

        output_sizes = [32, 32, 32, 256]

        self.quant_inp = qnn.QuantIdentity()

        # FINN 0.10: QuantConv2d MUST have bias=False !!

        # 1st layer
        self.conv1 = qnn.QuantConv2d(
            1,
            output_sizes[0],
            3,
            padding=1,
            bias=False,
            weight_bit_width=bit_width,
        )
        self.bn1 = nn.BatchNorm2d(output_sizes[0])
        self.relu1 = qnn.QuantReLU(bit_width=bit_width)

        # 2nd layer
        self.conv2 = qnn.QuantConv2d(
            output_sizes[0],
            output_sizes[1],
            3,
            padding=1,
            bias=False,
            weight_bit_width=bit_width,
        )
        self.bn2 = nn.BatchNorm2d(output_sizes[1])
        self.relu2 = qnn.QuantReLU(bit_width=bit_width)

        # 3rd conv
        self.conv3 = qnn.QuantConv2d(
            output_sizes[1],
            output_sizes[2],
            3,
            padding=1,
            bias=False,
            weight_bit_width=bit_width,
        )
        self.bn3 = nn.BatchNorm2d(output_sizes[2])
        self.relu3 = qnn.QuantReLU(bit_width=bit_width)

        self.flat = nn.Flatten()

        # Fully connected
        self.fc4 = qnn.QuantLinear(
            output_sizes[2] * np.prod(self.input_shape),
            output_sizes[3],
            bias=True,
            weight_bit_width=bit_width,
        )
        self.dropout4 = nn.Dropout(0.5)
        self.bn4 = nn.BatchNorm1d(output_sizes[3])
        self.relu4 = qnn.QuantReLU(bit_width=bit_width)

        self.fc5 = qnn.QuantLinear(
            output_sizes[3],
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, *self.input_shape))
        out = self.quant_inp(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.flat(out)
        out = self.relu4(self.bn4(self.dropout4(self.fc4(out))))
        logits = self.fc5(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = self.loss(y, y_true)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class EmagerSCNN(L.LightningModule):
    def __init__(self, input_shape):
        super().__init__()

        self.loss = etl.OnlineTripletLoss(
            0.2, etu.SemihardNegativeTripletSelector(0.2, False)
        )
        # self.loss = nn.TripletMarginLoss(margin=0.2)
        self.input_shape = input_shape

        output_sizes = [32, 32, 32]

        # 1st layer
        self.conv1 = nn.Conv2d(1, output_sizes[0], 3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_sizes[0])

        # 2nd layer
        self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(output_sizes[1])

        # 3rd conv
        self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 5, padding=2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(output_sizes[2])

        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, *self.input_shape))
        out = x
        out = self.bn1(self.relu1(self.conv1(out)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        embeddings = self(x)
        loss, n_triplets = self.loss(embeddings, y_true)
        self.log("train_loss", loss)
        self.log("triplets_generated", n_triplets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        # embeddings has shape (batchsize, embedding_size)
        embeddings = self(x)

        # class_embeddings has shape = (n_class, batchsize)
        class_embeddings = torch.zeros((6, embeddings.shape[1])).to(self.device)
        for i in range(len(class_embeddings)):
            class_idx = torch.where(y_true == i)[0]
            s = torch.sum(embeddings[class_idx], dim=0)
            class_embeddings[i] = s / len(s)

        embeddings = torch.transpose(embeddings, 0, 1) / torch.norm(embeddings, dim=1)
        preds = torch.matmul(class_embeddings, embeddings)
        acc = (preds.argmax(dim=0) == y_true).sum().item() / len(y_true)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    import emager_py.torch.dataset as etd
    import emager_py.utils as eutils
    import emager_py.transforms as etrans

    eutils.set_logging()
    train, val = etd.get_loocv_dataloaders(
        eutils.DATASETS_ROOT + "EMAGER/",
        "000",
        "001",
        9,
        transform=etrans.default_processing,
    )
    model = EmagerSCNN((4, 16))
    # model = EmagerCNN((4, 16), 6)
    trainer = L.Trainer(max_epochs=15)
    trainer.fit(model, train, val)
