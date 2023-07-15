"""
Instrument Recognition +
Transcription
"""

import torch
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

from omegaconf import OmegaConf
import pandas as pd
        
class Jointist(pl.LightningModule):
    def __init__(
        self,
        detection_model: pl.LightningModule,
        transcription_model: pl.LightningModule,
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
        self.transcription_model = transcription_model
        self.lr_lambda = lr_lambda
        self.cfg = cfg
        
    def training_step(self, batch, batch_idx):
        detection_loss = self.detection_model.training_step(batch, batch_idx, self)
        transcription_loss = self.transcription_model.training_step(batch, batch_idx, self)
        
        total_loss = transcription_loss+detection_loss
        self.log('Total_Loss/Train', total_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        outputs, detection_loss = self.detection_model.validation_step(batch, batch_idx, self)
        transcription_loss = self.transcription_model.validation_step(batch, batch_idx, self)
        total_loss = transcription_loss+detection_loss        
        self.log('Total_Loss/Valid', total_loss, on_step=False, on_epoch=True)
        
        return outputs, detection_loss
        
    def validation_epoch_end(self, outputs):
        detection_loss = self.detection_model.validation_epoch_end(outputs, self)

    def test_step(self, batch, batch_idx):
        plugin_idxs = self.detection_model.test_step(batch, batch_idx, self)        
        return self.transcription_model.test_step(batch, batch_idx, plugin_idxs, self)
        
    def test_epoch_end(self, outputs):
        self.transcription_model.test_epoch_end(outputs, self)
        
   
        
    def predict_step(self, batch, batch_idx):
        plugin_idxs = self.detection_model.predict_step(batch, batch_idx)
        self.transcription_model.predict_step(batch, batch_idx, plugin_idxs)        
        
        return plugin_idxs

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