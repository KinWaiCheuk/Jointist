import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl
from End2End.loss import HungarianMatcherv2, SetCriterion
from pytorch_lightning.utilities import rank_zero_only

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.metrics import (precision_recall_curve,
                             average_precision_score,
                             auc,
                             roc_auc_score,
                             precision_recall_fscore_support
                            )
import sys

        
class Hungarian(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
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
        self.network = network
        self.lr_lambda = lr_lambda
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.instrument_type = cfg.MIDI_MAPPING.type
        self.cfg = cfg
        
        # Setting up Hungarian Loss
        self.matcher = HungarianMatcherv2()
        self.criterion = SetCriterion(self.network.num_classes, matcher=self.matcher, weight_dict={'loss_ce':1},
                                      eos_coef=cfg.detection.model.eos_coef, losses=['labels'])
        
        
    def calculate_loss(self, batch):
        output = self.network(batch['waveform'])
        target = batch['instruments']
        target_count = batch['instruments'].sum(-1)
       
        target_list = []
        for sample in target:
            target_list.append({'labels': sample.nonzero().flatten()})  
            
        loss_dict, src_logits, target_classes = self.criterion(output, target_list)

        return loss_dict, src_logits, target_classes, output
        
        
    def training_step(self, batch, batch_idx, jointist=None):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch: dict, e.g., {
                'waveform': (batch_size, segment_samples),
                'onset_roll': (batch_size, frames_num, classes_num),
                ...
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # A hack to keep the logging functions work when wrapped with jointist
        if jointist:
            logger=jointist
        else:
            logger=self

        loss_dict, src_logits, target_classes, output = self.calculate_loss(batch)
        
        class_loss = loss_dict['loss_ce']
        
        loss = 0
        for key in loss_dict:
            if 'loss' in key:
                loss += loss_dict[key]
            
        for key in loss_dict:
            logger.log(f"{key}/Train", loss_dict[key], on_step=False, on_epoch=True)
        logger.log('Detection_Loss/Train', loss, on_step=False, on_epoch=True)

        if (self.current_epoch+1)%(self.trainer.check_val_every_n_epoch)==0: # log the label count only once        
            # For Evaluation
            predictions = []
            multihot_pred = []
            for i in output['pred_logits']:
                # Converting softmax into prediction via top1
                non_empty = i.argmax(-1)!=(self.network.num_classes-1)
                prediction = i[non_empty].argmax(-1).unique() # prevent duplication in the early stage 
                predictions.append(prediction)
                # Converting prediction into multihot
                multihot_pred.append(torch.nn.functional.one_hot(prediction,
                                     self.network.num_classes).sum(0))
            multihot_pred = torch.stack(multihot_pred)    

            if batch_idx==0:
                self.log_images(output['spec'].squeeze(1), f'Train/spectrogram', logger=logger)
                self._log_text(batch['instruments'], "Train/Labels", max_sentences=4, logger=logger)                
                self._log_pred_text(predictions, "Train/Prediction", max_sentences=4, logger=logger)

            output_batch = {
                'loss': loss,
                'instruments': batch['instruments'].detach(), # (B, num_classes)
                'multihot_pred': multihot_pred.detach(), # (B, num_classes)
                'src_logits':  src_logits.detach(), # (B, Q, num_classes)
                'target_classes': target_classes # list (B, num_instruments)
            }
            
            return output_batch
    
        return loss
    
    def calculate_epoch_end(self, outputs):
        """Calculating the same thing across train, val, and test set"""
        y_logits = []
        y_label = []
        y_pred_map = []
        y_label_map = []        
        for batch in outputs:
            y_logits.append(batch['src_logits']) # (B, Q, num_classes)
            y_label.append(batch['target_classes']) # list (B, num_instruments)
            y_pred_map.append(batch['multihot_pred']) # (B, num_classes)
            y_label_map.append(batch['instruments']) # (B, num_classes)       

        y_logits = torch.cat(y_logits)
        y_label = torch.cat(y_label)
        y_pred_map = torch.cat(y_pred_map)
        y_label_map = torch.cat(y_label_map)      

        # calculating PR-AUC and ROC-AUC
        metrics = self.get_accuracy(y_label, y_logits, OmegaConf.to_object(self.IX_TO_NAME))

        # calculating F1 scores       
        f1_dicts = {}
        non_zeros = (y_label_map.sum(0)!=0) # Finding non_zeros instruments                
        for key in [None, 'micro', 'macro', 'weighted', 'samples']:
            if key:
                # Only calculate the scores when the instrument exist in y_label
                _, _, f1, _ = precision_recall_fscore_support(y_label_map[:, non_zeros].cpu(), y_pred_map[:, non_zeros].cpu(), average=key, zero_division=0)               
                f1_dicts[key] = f1
            else:        
                _, _, f1, _ = precision_recall_fscore_support(y_label_map.cpu(), y_pred_map.cpu(), average=key, zero_division=0)
                f1_dicts['none'] = f1            

        # making bar plot
        pred_stat = {}
        label_stat = {}
        f1_stat = {}
        pred_counts = y_pred_map.sum(0)
        label_counts = y_label_map.sum(0)
        for (key, idx) in self.cfg.MIDI_MAPPING.NAME_TO_IX.items():
            pred_stat[key] = pred_counts[idx]
            label_stat[key] = label_counts[idx]
            f1_stat[key] = f1_dicts['none'][idx]
            
        results = {'f1_dicts': f1_dicts,
                   'metrics': metrics,
                   'pred_stat': pred_stat,
                   'label_stat': label_stat,
                   'f1_stat': f1_stat
                  }
        return results
    
    @rank_zero_only # To evaluate with only 1 GPU    
    def training_epoch_end(self, outputs, jointist=None):
        if (self.current_epoch+1)%(self.trainer.check_val_every_n_epoch)==0: # log the label count only once        
            # A hack to keep the logging functions work when wrapped with jointist
            if jointist:
                logger=jointist
            else:
                logger=self            

            results = self.calculate_epoch_end(outputs)

            _ = self.barplot(results['pred_stat'], 'train_pred_counts (log_e)', (4,12), 0.2, log=True)
            _ = self.barplot(results['label_stat'], 'train_label_counts (log_e)', (4,12), 0.2, log=True)
            logger.logger.experiment.add_figure(f"Train/F1 scores",
                                                self.barplot(results['f1_stat'], 'F1', (4,12), 0.05, log=False),
                                                global_step=self.current_epoch)

            for key in results['metrics']:# logging instrument-wise metrics
                for instrument in results['metrics'][key]:
                    logger.log(f"{key}_Train/{instrument}", results['metrics'][key][instrument], on_step=False, on_epoch=True, rank_zero_only=True)

            for key in results['f1_dicts']:
                if key!='none':
                    logger.log(f"F1_average_Train/{key}", results['f1_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
                else:
                    for idx, f1_score in enumerate(results['f1_dicts'][key]):
                        instrument = self.IX_TO_NAME[idx]
                        logger.log(f"F1_Train/{instrument}", results['f1_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)
    
    
    def validation_step(self, batch, batch_idx, jointist=None):
        # A hack to keep the logging functions work when wrapped with jointist
        if jointist:
            logger=jointist
        else:
            logger=self        
        
        metrics = {}     
        
        loss_dict, src_logits, target_classes, output = self.calculate_loss(batch)
        class_loss = loss_dict['loss_ce']
        
        loss = 0
        for key in loss_dict:
            if 'loss' in key:
                loss += loss_dict[key]
        
        for key in loss_dict:
            logger.log(f"{key}/Valid", loss_dict[key], on_step=False, on_epoch=True)
        metrics['Detection_Loss/Valid']=loss


        predictions = []
        multihot_pred = []
        for i in output['pred_logits']:
            # Converting softmax into prediction via top1
            non_empty = i.argmax(-1)!=(self.network.num_classes-1)
            prediction = i[non_empty].argmax(-1).unique() # prevent duplication in the early stage 
            predictions.append(prediction)
            # Converting prediction into multihot
            multihot_pred.append(torch.nn.functional.one_hot(prediction,
                                 self.network.num_classes).sum(0))
        multihot_pred = torch.stack(multihot_pred)    
        
        if batch_idx==0:
            if self.current_epoch==0:
                self._log_text(batch['instruments'], "Valid/Labels", max_sentences=4, logger=logger)
                self.log_images(output['spec'].squeeze(1), f'Valid/spectrogram', logger=logger)
            self._log_pred_text(predictions, "Valid/Prediction", max_sentences=4, logger=logger)
        logger.log_dict(metrics)
        
        output_batch = {
            'loss': loss,
            'instruments': batch['instruments'],
            'multihot_pred': multihot_pred,            
            'src_logits':  src_logits,
            'target_classes': target_classes
        }
        return output_batch 
        
    @rank_zero_only # To evaluate with only 1 GPU    
    def validation_epoch_end(self, outputs, jointist=None):
        # A hack to keep the logging functions work when wrapped with jointist
        if jointist:
            logger=jointist
        else:
            logger=self            
        results = self.calculate_epoch_end(outputs)
            
        _ = self.barplot(results['pred_stat'], 'valid_pred_counts (log_e)', (4,12), 0.2, log=True)
        _ = self.barplot(results['label_stat'], 'valid_label_counts (log_e)', (4,12), 0.2, log=True)
        logger.logger.experiment.add_figure(f"Valid/F1 scores",
                                            self.barplot(results['f1_stat'], 'F1', (4,12), 0.05, log=False),
                                            global_step=self.current_epoch)
        
        for key in results['metrics']:# logging instrument-wise metrics
            for instrument in results['metrics'][key]:
                logger.log(f"{key}_Valid/{instrument}", results['metrics'][key][instrument], on_step=False, on_epoch=True, rank_zero_only=True)
                
        for key in results['f1_dicts']:
            if key!='none':
                logger.log(f"F1_average_Valid/{key}", results['f1_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
            else:
                for idx, f1_score in enumerate(results['f1_dicts'][key]):
                    instrument = self.IX_TO_NAME[idx]
                    logger.log(f"F1_Valid/{instrument}", results['f1_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)
                      

    def test_step(self, batch, batch_idx, jointist=None):
        # A hack to keep the logging functions work when wrapped with jointist
        if jointist:
            logger=jointist
        else:
            logger=self        
        
        metrics = {}     
        
        loss_dict, src_logits, target_classes, output = self.calculate_loss(batch)
        
        class_loss = loss_dict['loss_ce']
        
        loss = 0
        for key in loss_dict:
            if 'loss' in key:
                loss += loss_dict[key]
        
        for key in loss_dict:
            logger.log(f"{key}/Test", loss_dict[key], on_step=False, on_epoch=True)
        metrics['Detection_Loss/Test']=loss


        predictions = []
        multihot_pred = []
        for i in output['pred_logits']:
            # Converting softmax into prediction via top1
            non_empty = i.argmax(-1)!=(self.network.num_classes-1)
            prediction = i[non_empty].argmax(-1).unique() # prevent duplication in the early stage 
            predictions.append(prediction)
            # Converting prediction into multihot
            multihot_pred.append(torch.nn.functional.one_hot(prediction,
                                 self.network.num_classes).sum(0))
        multihot_pred = torch.stack(multihot_pred)    
        
        if batch_idx==0:
            if self.current_epoch==0:
                self._log_text(batch['instruments'], "Test/Labels", max_sentences=4, logger=logger)
                self.log_images(output['spec'].squeeze(1), f'Test/spectrogram', logger=logger)
            self._log_pred_text(predictions, "Test/Prediction", max_sentences=4, logger=logger)
        logger.log_dict(metrics)
        
        output_batch = {
            'loss': loss,
            'instruments': batch['instruments'],
            'multihot_pred': multihot_pred,            
            'src_logits':  src_logits,
            'target_classes': target_classes
        }
        return output_batch 

    @rank_zero_only # To evaluate with only 1 GPU  
    def test_epoch_end(self, outputs, jointist=None):
        # A hack to keep the logging functions work when wrapped with jointist
        if jointist:
            logger=jointist
        else:
            logger=self            
        results = self.calculate_epoch_end(outputs)
            
        _ = self.barplot(results['pred_stat'], 'test_pred_counts (log_e)', (4,12), 0.2, log=True)
        _ = self.barplot(results['label_stat'], 'test_label_counts (log_e)', (4,12), 0.2, log=True)
        logger.logger.experiment.add_figure(f"Test/F1 scores",
                                            self.barplot(results['f1_stat'], 'F1', (4,12), 0.05, log=False),
                                            global_step=self.current_epoch)
        
        for key in results['metrics']:# logging instrument-wise metrics
            for instrument in results['metrics'][key]:
                logger.log(f"{key}_Test/{instrument}", results['metrics'][key][instrument], on_step=False, on_epoch=True, rank_zero_only=True)
                
        for key in results['f1_dicts']:
            if key!='none':
                logger.log(f"F1_average_Test/{key}", results['f1_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
            else:
                for idx, f1_score in enumerate(results['f1_dicts'][key]):
                    instrument = self.IX_TO_NAME[idx]
                    logger.log(f"F1_Test/{instrument}", results['f1_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)
   
        
    def predict_step(self, batch, batch_idx):
        loss_dict, src_logits, target_classes, output = self.calculate_loss(batch)
        
        # Converting softmax into prediction via top1
        predictions = []
        multihot_pred = []
        for i in output['pred_logits']:
            non_empty = i.argmax(-1)!=(self.network.num_classes-1)
            predictions.append(i[non_empty].argmax(-1))
            # Converting prediction into multihot
            multihot_pred.append(torch.nn.functional.one_hot(i[non_empty].argmax(-1),
                                 self.network.num_classes).sum(0))
        multihot_pred = torch.stack(multihot_pred)

    
    
        plugin_list = []
        for sample in batch['instruments']:
            plugin_str = ''
            for i in sample.nonzero(as_tuple=False):
                plugin_str = plugin_str + self.IX_TO_NAME[i.item()] + ', '
            plugin_list.append(plugin_str)        
        s_label = pd.Series(plugin_list, name="plugin_names")     
        
        
        plugin_list = []
        for sample in predictions:
            plugin_str = ''
            for i in sample:
                plugin_str = plugin_str + self.IX_TO_NAME[i.item()] + ', '
            plugin_list.append(plugin_str)        
        s_pred = pd.Series(plugin_list, name="plugin_names")        

        return {
            'src_logits':  src_logits,
            'instruments': batch['instruments'],
            'predictions': predictions,
            'multihot_pred': multihot_pred,
            'target_classes': target_classes,
            's_label': s_label,
            's_pred': s_pred,
            'hdf5_name': batch['hdf5_name']
        }

    def configure_optimizers(self):
        r"""Configure optimizer."""
        optimizer = optim.Adam(
            self.network.parameters(),
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
    
    def _log_text(self, multihot_vector, tag, max_sentences, logger):       
        plugin_list = []
        for sample in multihot_vector:
            plugin_str = ''
            for i in sample.nonzero(as_tuple=False):
                plugin_str = plugin_str + self.IX_TO_NAME[i.item()] + ', '
            plugin_list.append(plugin_str)        
        s = pd.Series(plugin_list, name="plugin_names")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.current_epoch)
        
    
    def _log_pred_text(self, predicted_instruments, tag, max_sentences, logger):       
        plugin_list = []
        for sample in predicted_instruments:
            plugin_str = ''
            for i in sample:
                plugin_str = plugin_str + self.IX_TO_NAME[i.item()] + ', '
            plugin_list.append(plugin_str)        
        s = pd.Series(plugin_list, name="plugin_names")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.current_epoch)        
        
    def log_images(self, tensor, key, logger):    
        fig, axes = plt.subplots(2,2)
        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(tensor[idx].cpu().detach().t(), aspect='auto', origin='lower', cmap='jet')    
        logger.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)
        
    def barplot(self, metric, title, figsize=(4,12), minor_interval=0.2, log=False):
        fig, ax = plt.subplots(1,1, figsize=figsize)
        metric = {k: v for k, v in sorted(metric.items(), key=lambda item: item[1])}
        xlabels = list(metric.keys())
        values = list(metric.values())
        if log:
            values = np.log(values)
        ax.barh(xlabels, values)
        ax.tick_params(labeltop=True, labelright=False)
        ax.xaxis.grid(True, which='minor')
        ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
        ax.set_ylim([-1,len(xlabels)])
        ax.set_title(title)
        ax.grid(axis='x')
        ax.grid(b=True, which='minor', linestyle='--')
        fig.savefig(f'{title}.png', bbox_inches='tight')
        fig.tight_layout() # prevent edge from missing
#         fig.set_tight_layout(True)
        return fig
        
        
    def get_accuracy(self, target_classes, src_logits, idx2name_map):
        precision = dict()
        recall = dict()
        average_precision = dict()
        PR_AUC = dict()
        ROC_AUC = dict()
#         idx2name_map[len(idx2name_map)]="empty" # no need this one anymore
#         mask = src_logits.argmax(-1)!=(src_logits.shape[-1]-1)

        mask = torch.logical_or(
            src_logits.argmax(-1)!=(src_logits.shape[-1]-1),
            target_classes!=(src_logits.shape[-1]-1)
            ) # (B, Q, num_classes)
     

        y_score = torch.softmax(src_logits[mask], -1).detach().cpu() # (B*Q-msaked, num_classes)
        Y_test = target_classes[mask].cpu()

        for idx, i in enumerate(np.unique(Y_test)):
            p, r, _ = precision_recall_curve(Y_test==i, y_score[:, i])
            AP = average_precision_score(Y_test==i, y_score[:, i])
            roc_auc = roc_auc_score(Y_test==i, y_score[:, i])
            if np.isnan(AP):
                break        

            precision[f"{idx2name_map[i]}_{i}"] = p
            recall[f"{idx2name_map[i]}_{i}"] = r
            average_precision[f"{idx2name_map[i]}_{i}"] = AP
            PR_AUC[f"{idx2name_map[i]}_{i}"] = auc(r,p)
            ROC_AUC[f"{idx2name_map[i]}_{i}"] = roc_auc

        return {
    #             'precision': precision,
    #             'recall': recall,
            'average_precision': average_precision,
            'PR_AUC': PR_AUC,
            'ROC_AUC': ROC_AUC
            }
    
    