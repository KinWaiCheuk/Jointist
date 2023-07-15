"""
Instrument Recognition +
Transcription +
Music Source Separation
"""

import torch
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

from omegaconf import OmegaConf
import pandas as pd
        
class Jointist_SS(pl.LightningModule):
    def __init__(
        self,
        detection_model: pl.LightningModule,
        tseparation_model: pl.LightningModule,
        lr_lambda,
        cfg
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            network: nn.Module
            loss_function: func
            learning_rate, float, e.g., 1e-3
            lr_lambda: func
        """
        super().__init__()
        self.detection_model = detection_model
        self.tseparation_model = tseparation_model
        self.lr_lambda = lr_lambda
        self.cfg = cfg
   
        
    def predict_step(self, batch, batch_idx):
        plugin_idxs = self.detection_model.predict_step(batch, batch_idx)
        self.tseparation_model.predict_step(batch, batch_idx, plugin_idxs)
        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        optimizer = optim.Adam(
            list(self.transcription_model.parameters()) + list(self.detection_model.parameters()),
            **self.cfg.detection.model.optimizer,
        )

        if self.cfg.scheduler.type=="MultiStepLR":
            scheduler = {
                'scheduler': MultiStepLR(optimizer,
                                         milestones=list(self.cfg.scheduler.milestones),
                                         gamma=self.cfg.scheduler.gamma),          
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self.cfg.scheduler.type=="LambdaLR":
            scheduler = {
                'scheduler': LambdaLR(optimizer, self.lr_lambda),
                'interval': 'step',
                'frequency': 1,
            }
            

        return [optimizer], [scheduler]