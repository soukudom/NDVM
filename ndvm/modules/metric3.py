"""
    Dataset Class Similarity Calculation
"""

import math
import multiprocessing

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from core import AbstractMetric

class Similarity(AbstractMetric):
    def __init__(self, dataset, label, multiclass, verbose):
        self.dataset = dataset
        self.label = label
        self.verbose = verbose
        self.MULTICLASS = multiclass

    def get_name(self):
        return "Similarity"
    
    def get_details(self):
        pass

    def run_evaluation(self):
        return self.class_similarity(self.dataset, self.dataset.drop(columns=[self.label]).columns, self.label)

    def class_similarity(self, df, feature_cols, label_col, base_class=None, max_epochs=50):
        """Calculates class similarity metric (M_3).

        Args:
            df (pandas.DataFrame): Input dataframe.
            feature_cols ([str]): Feature column names.
            label_col (str): Label column name.
            base_class (str, optional): Value of base class. Defaults to None.
            max_epochs (int, optional): Epochs to train autoencoder. Defaults to 50.

        Returns:
            dict: Dictionary containing metric report.
        """
        df = df.copy()

        # retype
        df[label_col] = df[label_col].astype(str)

        # base class = majority class
        if not base_class:
            base_class = df[label_col].value_counts().index[0]

        # scale
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        # train base autoencoder
        model = self.train_ae(df[df[label_col] == base_class][feature_cols], max_epochs=max_epochs)
        model.eval()

        result = {}

        # get reconstruction error
        for label in df[label_col].unique():
            tensor_label = torch.tensor(df[df[label_col] == label][feature_cols].to_numpy().astype(np.float32))

            z = model.encoder(tensor_label)
            x_hat = torch.sigmoid(model.decoder(z))

            loss = F.l1_loss(x_hat, tensor_label)

            result[label] = {
                "l1_loss": loss.item(),
            }

        # relative errors
        all_count = 0
        all_relative = 0

        for label in df[label_col].unique():
            result[label]["relative_loss"] = result[label]["l1_loss"] / result[base_class]["l1_loss"]

            # skip base class
            if label == base_class:
                continue

            label_count = (df[label_col] == label).sum()

            all_count += label_count
            all_relative += label_count * result[label]["relative_loss"]

        return {
            "metric": all_relative / all_count,
            "detail": result,
        }


    def train_ae(self, df, max_epochs):
        """Train autoencoder for metric evaluation.

        Args:
            df (pandas.DataFrame): Input dataframe.
            max_epochs (int): Epochs to train the autoencoder.

        Returns:
            SimpleAutoencoder: Trained model.
        """
        train_size = int(0.8 * len(df))
        valid_size = len(df) - train_size

        train, valid = torch.utils.data.random_split(df.to_numpy().astype(np.float32), [train_size, valid_size])

        nproc = multiprocessing.cpu_count()

        train_dataloader = torch.utils.data.DataLoader(
            train,
            batch_size=32,
            shuffle=True,
            num_workers=nproc,
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid,
            batch_size=32,
            shuffle=False,
            num_workers=nproc,
        )

        # hidden layer prep
        #
        # in evaluated datasets:
        #   - 42 [16] 8
        #   - 79 [32] 16
        #   - 24 [8] 4

        input_size = len(df.columns)
        hidden_size_1 = 2 ** (math.floor(math.log2(input_size)) - 1)

        model = SimpleAutoencoder(input_size, [hidden_size_1], hidden_size_1 // 2)

        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        return model


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, input: int, hidden: list[int], latent: int):
        """Simple autoencoder with linear layers and ReLU activations.

        Args:
            input (int): number of input features
            hidden (list[int]): list of hidden layer sizes
            latent (int): number of latent features

        Examples::
            >>> SimpleAutoencoder(32, [16, 8], 4)
            SimpleAutoencoder(
                (encoder): Sequential(
                    (0): Linear(in_features=32, out_features=16, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=16, out_features=8, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=8, out_features=4, bias=True)
                )
                (decoder): Sequential(
                    (0): Linear(in_features=4, out_features=8, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=8, out_features=16, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=16, out_features=32, bias=True)
                )
            )
        """
        super().__init__()

        # encoder setup
        encoder_layers = []

        i_in = input

        for i_out in hidden:
            encoder_layers.append(nn.Linear(i_in, i_out))
            encoder_layers.append(nn.ReLU())
            i_in = i_out

        encoder_layers.append(nn.Linear(i_in, latent))

        self.encoder = nn.Sequential(*encoder_layers)

        # decoder setup
        decoder_layers = []

        i_in = latent

        for i_out in reversed(hidden):
            decoder_layers.append(nn.Linear(i_in, i_out))
            decoder_layers.append(nn.ReLU())
            i_in = i_out

        decoder_layers.append(nn.Linear(i_in, input))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]

        z = self.encoder(x)
        x_hat = torch.sigmoid(self.decoder(z))

        loss = F.binary_cross_entropy(x_hat, x)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]

        z = self.encoder(x)
        x_hat = torch.sigmoid(self.decoder(z))

        loss = F.l1_loss(x_hat, x)

        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
