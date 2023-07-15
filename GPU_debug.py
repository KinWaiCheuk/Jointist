import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch.optim as optim

import matplotlib.pyplot as plt


X, Y = make_blobs(10000,1000,centers=10, cluster_std=10)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),torch.from_numpy(y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=2)


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(100, 256, bidirectional=True)
        self.classifier = nn.Linear(256*2*10,10)

    def forward(self, x):
        x, _ = self.lstm(x.view(-1,10,100))
        x = self.classifier(x.flatten(1))
        return x


    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.cross_entropy(pred, batch[1])
        return loss

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters())


model = Model()

trainer = pl.Trainer(max_epochs=99999, gpus=2, accelerator="ddp")


trainer.fit(model, trainloader)
# check if bin 0-20 has changed
