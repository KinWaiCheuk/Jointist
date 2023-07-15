import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

from omegaconf import OmegaConf
import pandas as pd

# for applying threshold on outputroll
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        # ctx is the context object that can be called in backward
        # in DANN we use it to save the reversal scaler lambda
        # in this case, we don't need to use it
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Since we have two inputs during forward()
        # There would be grad for both x and threshold
        # Therefore we need to return a grad for each
        # But we don't need grad for threshold, so make it None        
        return F.hardtanh(grad_output), None
        
class TSeparation(pl.LightningModule):
    def __init__(
        self,
        transcription_model: pl.LightningModule,
        separation_model: pl.LightningModule,
        batch_data_preprocessor,
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
        self.transcription_model = transcription_model
        self.separation_model = separation_model        
        self.lr_lambda = lr_lambda
        self.batch_data_preprocessor = batch_data_preprocessor
        self.cfg = cfg
        
    def training_step(self, batch, batch_idx):
        batch = self.batch_data_preprocessor(batch)
        transcription_output = self.transcription_model.training_step(batch, batch_idx, self)
        transcription_loss = transcription_output['loss']
        outputs = transcription_output['outputs']
        if self.cfg.straight_through==True:
            outputs['frame_output'] = STE.apply(outputs['frame_output'], self.cfg.transcription.evaluation.frame_threshold)
        separation_loss = self.separation_model.training_step(batch, batch_idx, outputs, self)
        
        total_loss = transcription_loss + separation_loss
        self.log('Total_Loss/Train', total_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        batch = self.batch_data_preprocessor(batch)        
        transcription_loss, outputs = self.transcription_model.validation_step(batch, batch_idx, self)
        if self.cfg.straight_through==True:
            outputs['frame_output'] = STE.apply(outputs['frame_output'], self.cfg.transcription.evaluation.frame_threshold)
        separation_loss = self.separation_model.validation_step(batch, batch_idx, outputs, self)
        
        total_loss = transcription_loss + separation_loss['Separation/Valid/Loss']
        self.log('Total_Loss/Valid', total_loss, on_step=False, on_epoch=True)
        
        return outputs

    def test_step(self, batch, batch_idx):
        # TODO: Update it for Jointist
        _, _, output_dict = self.transcription_model.test_step(batch,batch_idx, None, False, self)
        if self.cfg.straight_through==True:
            output_dict['frame_output'] = STE.apply(output_dict['frame_output'], self.cfg.transcription.evaluation.frame_threshold)
        
        sdr_dict = self.separation_model.test_step(batch,batch_idx, output_dict, self)

        return sdr_dict
        
    def test_epoch_end(self, outputs):
        self.separation_model.test_epoch_end(outputs, self)
        
   
        
    def predict_step(self, batch, batch_idx, plugin_idxs):
        output_dict = self.transcription_model.predict_step(batch, batch_idx, plugin_idxs, True)     
        self.separation_model.predict_step(batch, batch_idx, output_dict, plugin_idxs)      

    def configure_optimizers(self):
        r"""Configure optimizer."""
        optimizer = optim.Adam(
            list(self.transcription_model.parameters()) + list(self.separation_model.parameters()),
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
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