import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

# from End2End.inference_instruments_filter import RegressionPostProcessor, OnsetFramePostProcessor
import End2End.inference_instruments_filter as PostProcessor
from End2End.constants import SAMPLE_RATE
from End2End.transcription_utils import (
                                        postprocess_probabilities_to_midi_events,
                                        predict_probabilities,
                                        write_midi_events_to_midi_file,
                                        predict_probabilities_baseline
                                        )
from End2End.tasks.transcription.utils import (calculate_mean_std,
                                               calculate_intrumentwise_statistics,
                                               evaluate_F1,
                                               evaluate_flat_F1,
                                               piecewise_evaluation,
                                               get_flat_average,
                                               barplot
                                              )
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

import sys


class Transcription(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_function,
        lr_lambda,
        batch_data_preprocessor,
        cfg
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            network: nn.Module
            list_programs: list of list, e.g., [
                ['0', '16', '33', '48'],    # Slakh2100 dataset
                ['percussion'],             # Groove dataset
            ]
            loss_function: func
            learning_rate, float, e.g., 1e-3
            lr_lambda: func
        """
        super().__init__()
        # TODO: Try to add save_parameters
        self.network = network
        self.loss_function = loss_function
        self.learning_rate = cfg.lr
        self.lr_lambda = lr_lambda
        self.classes_num = cfg.transcription.model.args.classes_num
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.instrument_type = cfg.MIDI_MAPPING.type
        self.cfg = cfg
        
        self.seg_batch_size = cfg.transcription.evaluation.seg_batch_size
        self.segment_samples = cfg.segment_seconds*SAMPLE_RATE
        if hasattr(cfg.datamodule, 'dataset_cfg'): # this is for training and testing
            self.test_segment_size = cfg.datamodule.dataset_cfg.test.segment_seconds
        elif hasattr(cfg.datamodule, 'dataloader_cfg'): # this is for inferencing (pred_transcription.py)
            self.test_segment_size = cfg.segment_seconds            
#         self.evaluation_output_path = cfg.evaluation.output_path
        self.evaluation_output_path = os.path.join(os.getcwd(), 'MIDI_output')
        if cfg.datamodule.type == 'slakh':
            self.pkl_dir = cfg.datamodule.pkl_dir    
        os.makedirs(self.evaluation_output_path, exist_ok=True)
        
        self.frame_threshold = cfg.transcription.postprocessor.args.frame_threshold
        self.onset_threshold = cfg.transcription.postprocessor.args.onset_threshold
        
        self.batch_data_preprocessor = batch_data_preprocessor
#         self.post_processor = RegressionPostProcessor(**cfg.transcription.postprocessor)
        self.post_processor = getattr(PostProcessor, cfg.transcription.postprocessor.type)(**cfg.transcription.postprocessor.args)
        

        

        # self.all_programs = list(itertools.chain(*list_programs))
        # E.g., [['0', '16', '33', '48'], ['percussion']] -> ['0', '16', '33', '48', 'percussion']

    def training_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self          
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: dict, e.g., {
                'waveform': (batch_size, segment_samples),
                '0_reg_onset_roll': (batch_size, frames_num, classes_num),
                'percussion_reg_onset_roll': (batch_size, frames_num, classes_num),
                ...
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # from IPython import embed; embed(using=False); os._exit(0)
        if self.batch_data_preprocessor:
            batch = self.batch_data_preprocessor(batch)
        # from IPython import embed; embed(using=False); os._exit(0)


        # Forward.
        # use the first batch and discard the rest to minimize code modification
        # TODO: develop a better sampling method, or even use all batches (need a for loop)
        target_dict = batch['target_dict']
        outputs = self.network(batch['waveforms'], batch['conditions'])
            # E.g., {
            #     'reg_onset_output': (12, 1001, 88 * 5),
            #     'reg_offset_output': (12, 1001, 88 * 5),
            #     'frame_output': (12, 1001, 88 * 5),
            #     ...
            # }

            # from IPython import embed; embed(using=False); os._exit(0)               
        loss = self.loss_function(self.network, outputs, target_dict)
        
        logger.log('Transcription_Loss/Train', loss)
        
        output_dict = {'loss': loss,
                       'outputs': outputs}
        return output_dict # pl only supports single return, be it tensor or dict
    
    def validation_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self             
        valid_metrics = {}      
        
        # Logging loss
        if self.batch_data_preprocessor:
            batch = self.batch_data_preprocessor(batch)
        target_dict = batch['target_dict']
        outputs = self.network(batch['waveforms'], batch['conditions'])

        # Calculate loss.
        if batch_idx<4:
            for key in target_dict:
                if self.current_epoch==0:
#                     self.log_images(outputs['spec'].squeeze(1), f'Valid/spectrogram')
                    self.log_images(target_dict[key].squeeze(1), batch['conditions'], f'Valid/{key}', batch_idx, logger)
            for key in outputs:
                self.log_images(outputs[key].squeeze(1), batch['conditions'], f'Valid/{key}', batch_idx, logger)
                
        loss = self.loss_function(self.network, outputs, target_dict) # self.network is not needed
        valid_metrics['Transcription_Loss/Valid']=loss
           
        
        logger.log_dict(valid_metrics)
        
        return loss, outputs
        
    def test_step(self, batch, batch_idx, plugin_ids=None, export=True, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self                 
        conditions = torch.eye(self.plugin_labels_num) # assume all instruments are presented
        conditions = conditions[:-1] # remove the Empty condition            

        if type(plugin_ids)==torch.Tensor:
            if len(plugin_ids) == 0: # When there is no instrument detected
                plugin_ids = torch.arange(self.plugin_labels_num-1)
                conditions = conditions[plugin_ids]
            else:
                conditions = conditions[plugin_ids]
        elif plugin_ids==None:                  
            plugin_ids = torch.where(batch['instruments'][0]==1)[0] 
            conditions = torch.zeros((len(plugin_ids), self.plugin_labels_num), device=plugin_ids.device)
    #         conditions.zero_()
            conditions.scatter_(1, plugin_ids.view(-1,1), 1)
        else:
            raise ValueError(f"plugin_ids has an unknown type: {type(plugin_ids)}")

            


        audio = batch['waveform']
        trackname = batch['hdf5_name'][0]

        midi_events = {}
        output_dict = {}
        framewise_dict = {}
        flat_output_dict = {}
        flattarget_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []
            flat_output_dict[key] = []
            flattarget_dict[key] = []

        # --- 1. Predict probabilities ---
#         print('--- 1. Predict probabilities ---')        
        num_instruments = len(conditions)
        framewise_dict[trackname] = {}
        flat_framewise_dict = {}        
        flatroll_list = []
        for condition in conditions:
            if self.test_segment_size!=None:              
                _output_dict = self.network(batch['waveform'], condition.unsqueeze(0))
                # _output_dict['frame_output'] = (B, T, F)
            elif self.test_segment_size==None:
#             print(f"Predicting {self.IX_TO_NAME[condition.argmax().item()]} ({idx}/{num_instruments})", end='\r')
                _output_dict = predict_probabilities(self.network, audio.squeeze(0), condition, self.segment_samples, self.seg_batch_size)
            else:
                raise ValueError(f"{self.test_segment_size=} is not defined")

            # calculating framewise score
            idx = condition.argmax().item()
            try:
                target_roll = batch['target_dict'][0][self.IX_TO_NAME[idx]]['frame_roll']
                pred_roll = _output_dict['frame_output']        

                reg_onset_roll = _output_dict['reg_onset_output']
                timesteps = min(len(target_roll), len(pred_roll))
                if export:               
                    frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(target_roll[:timesteps].flatten(),
                                                                                    pred_roll[:timesteps].flatten()>self.frame_threshold,
                                                                                    average='binary')

                    framewise_dict[trackname][self.IX_TO_NAME[idx]] = {
                        'precision': frame_p,
                        'recall': frame_r,
                        'f1': frame_f1}
                else:
                    framewise_dict = {}
                for key in ['reg_onset_output', 'frame_output']:
                    output_dict[key].append(_output_dict[key]) # (timesteps, 88)                    


                if export:
                    if idx!=38:     
                        # Flattening different instruments into a single flat roll
                        if 'flatroll' in locals():
                            flatroll = torch.logical_or(flatroll, pred_roll[:timesteps]>self.frame_threshold)
                            flattarget_roll = np.logical_or(flattarget_roll, target_roll[:timesteps])                
                            flatreg_onset_roll = torch.logical_or(flatreg_onset_roll, reg_onset_roll[:timesteps]>self.onset_threshold)
                        else:
                            flatroll = (pred_roll[:timesteps]>self.frame_threshold)
                            flattarget_roll = torch.from_numpy(target_roll[:timesteps])
                            flatreg_onset_roll = reg_onset_roll[:timesteps]>self.onset_threshold
            except:
                pass


        if export:
            # Preparing flat rolls
            flat_output_dict['frame_output'].append(flatroll)
            flat_output_dict['reg_onset_output'].append(flatreg_onset_roll)  
            flattarget_dict['frame_output'].append(flattarget_roll)
            flattarget_dict['reg_onset_output'].append(flattarget_roll)


            flat_frame_p, flat_frame_r, flat_frame_f1, _ = precision_recall_fscore_support(flattarget_roll[:timesteps].flatten(),
                                                                            flatroll[:timesteps].flatten(),
                                                                            average='binary')                

            flat_framewise_dict[trackname]  = {
                'precision': flat_frame_p,
                'recall': flat_frame_r,
                'f1': flat_frame_f1}
        else:
            flat_framewise_dict = {}

        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = torch.cat(output_dict[key], axis=-1).float() # (timesteps, 88*num_conditions)
            if export:
                flat_output_dict[key] = torch.cat(flat_output_dict[key], axis=-1).float()
                flattarget_dict[key] = torch.cat(flattarget_dict[key], axis=-1).float()

#         --- 2. Postprocess probabilities to MIDI events ---
#         print('--- 2. Postprocess probabilities to MIDI events ---')

        if export:
            midi_events = postprocess_probabilities_to_midi_events(output_dict, plugin_ids, self.IX_TO_NAME, self.classes_num, self.post_processor)
            flat_midi_events = postprocess_probabilities_to_midi_events(flat_output_dict, plugin_ids[:1], self.IX_TO_NAME, self.classes_num, self.post_processor)
            flat_midi_labels = postprocess_probabilities_to_midi_events(flattarget_dict, plugin_ids[:1], self.IX_TO_NAME, self.classes_num, self.post_processor)        
            pickle.dump(midi_events, open(os.path.join(self.evaluation_output_path,f"{trackname}.pkl"), 'wb'))
            pickle.dump(flat_midi_events, open(os.path.join(self.evaluation_output_path,f"{trackname}.flat_pkl"), 'wb'))
            pickle.dump(flat_midi_labels, open(os.path.join(self.evaluation_output_path,f"{trackname}.label_pkl"), 'wb'))
            ## TODO: try to evaluate flat notewise score using the dump files

            # --- 3. Write MIDI events to audio ---
    #         print('--- 3. Write MIDI events to audio ---')

            midi_path = os.path.join(self.evaluation_output_path,f"{trackname}.mid")
            write_midi_events_to_midi_file(midi_events, midi_path, self.instrument_type)
        return framewise_dict, flat_framewise_dict, output_dict

    def test_epoch_end(self, outputs, jointist=None):
        """
          outputs = [
                     {dict1}, {dict2},
                    {dict1}, {dict2},
                     ...
                     {dict1}, {dict2}
                    ]
        """
        
        framewise_dict = {'framewise': {},
                          'flat_framewise': {}}
        for idx, output_dict in enumerate(outputs):
            trackname = list(output_dict[0].keys())[0] # key here is the trackname
            framewise_dict['framewise'][trackname] = outputs[idx][0][trackname]
            framewise_dict['flat_framewise'][trackname] = outputs[idx][1][trackname]
    
        if jointist:
            logger=jointist
        else:
            logger=self                     
        pred_path = self.evaluation_output_path
        label_path = os.path.join(self.pkl_dir, 'test')       
        
        notewise_dict = evaluate_F1(pred_path, label_path)
        
        flat_notewise_dict = evaluate_flat_F1(pred_path)        
        pickle.dump(notewise_dict, open("notewise_dict.pkl", 'wb')) # saving the notewise_dict as pickle file
        pickle.dump(flat_notewise_dict, open("flat_notewise_dict.pkl", 'wb')) # saving the notewise_dict as pickle file        
        
        # computing piecewise metrics
        piecewise_frame_f1 = np.mean(piecewise_evaluation(framewise_dict, 'framewise', 'f1'))
        piecewise_note_f1 = np.mean(piecewise_evaluation(notewise_dict, 'note', 'f1'))
        piecewise_note_w_off_f1 = np.mean(piecewise_evaluation(notewise_dict, 'note_w_off', 'f1'))        
        
        # computing instrumentwise metrics
        
        ## frame-wise metrics
        instrument_wise_precision, instrument_wise_recall, instrument_wise_F1 = calculate_intrumentwise_statistics(notewise_dict, 'note')
        instrument_wise_precision_woff, instrument_wise_recall_woff, instrument_wise_F1_woff = calculate_intrumentwise_statistics(notewise_dict, 'note_w_off')
        instrument_wise_precision_frame, instrument_wise_recall_frame, instrument_wise_F1_frame = calculate_intrumentwise_statistics(framewise_dict, 'framewise')        
        
        ## calculate note-wise metrics
        precision_mean, precision_std = calculate_mean_std(instrument_wise_precision)
        recall_mean, recall_std = calculate_mean_std(instrument_wise_recall)
        F1_mean, F1_std = calculate_mean_std(instrument_wise_F1)
        
        ## calculate note-wise-with-offset metrics        
        precision_mean_woff, precision_std_woff = calculate_mean_std(instrument_wise_precision_woff)
        recall_mean_woff, recall_std_woff = calculate_mean_std(instrument_wise_recall_woff)
        F1_mean_woff, F1_std_woff = calculate_mean_std(instrument_wise_F1_woff)
        
        
        # computing mean and std
        ## for instrumentwise:
        precision_mean_frame, precision_std_frame = calculate_mean_std(instrument_wise_precision_frame)
        recall_mean_frame, recall_std_frame = calculate_mean_std(instrument_wise_recall_frame)
        F1_mean_frame, F1_std_frame = calculate_mean_std(instrument_wise_F1_frame)        
        
        ## for flat:
#         torch.save(framewise_dict, 'framewise_dict.pt')
        flat_p_mean_frame, flat_r_mean_frame, flat_f1_mean_frame = get_flat_average(framewise_dict, 'flat_framewise')
        flat_p_mean_note, flat_r_mean_note, flat_f1_mean_note = get_flat_average(flat_notewise_dict, 'note')
        flat_p_mean_note_w_off, flat_r_mean_note_w_off, flat_f1_mean_note_w_off = get_flat_average(flat_notewise_dict, 'note_w_off')        
#         summary = [[scores['precision'], scores['recall'], scores['f1']] for trackname, scores in framewise_dict['flat_framewise'].items()]
#         flat_p_mean_frame, flat_r_mean_frame, flat_f1_mean_frame = np.mean(summary, axis=0)  
        
        global_mean, fig_notef1 = barplot(F1_mean, 'Notewise F1 scores', figsize=(4,36))
        global_mean_woff, fig_noteoffsetf1 = barplot(F1_mean_woff, 'Notewise-offset F1 scores', figsize=(4,36))
        global_mean_frame, fig_frame = barplot(F1_mean_frame, 'Framewise F1 scores', figsize=(4,36))        
        
        logger.logger.experiment.add_figure(f"Test/Notewise F1 scores",
                                            fig_notef1,
                                            global_step=self.current_epoch)
        logger.logger.experiment.add_figure(f"Test/Notewise offset F1 scores",
                                            fig_noteoffsetf1,
                                            global_step=self.current_epoch)
        logger.logger.experiment.add_figure(f"Test/Framewise F1 scores",
                                            fig_frame,
                                            global_step=self.current_epoch)
        
        logger.log(f"Test/notewise", global_mean, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/notewise_w_offset", global_mean_woff, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Framewise", global_mean_frame, on_step=False, on_epoch=True, rank_zero_only=True)
                                   
        logger.log(f"Test/piecewise_notewise", piecewise_note_f1, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/piecewise_notewise_w_offset", piecewise_note_w_off_f1, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/piecewise_Framewise", piecewise_frame_f1, on_step=False, on_epoch=True, rank_zero_only=True)                                   
        
        
        logger.log(f"Test/Flat_Framewise", flat_f1_mean_frame, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Flat_notewise", flat_f1_mean_note, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Flat_notewise_w_offset", flat_f1_mean_note_w_off, on_step=False, on_epoch=True, rank_zero_only=True)
        
    def predict_step(self, batch, batch_idx, plugin_ids=None, return_roll=False):
        
        conditions = torch.eye(self.plugin_labels_num) # assume all instruments are presented
        conditions = conditions[:-1] # remove the Empty condition

        if type(plugin_ids)==torch.Tensor:
            if len(plugin_ids) == 0: # When there is no instrument detected
                plugin_ids = torch.arange(self.plugin_labels_num-1)
                conditions = conditions[plugin_ids]
            else:
                conditions = conditions[plugin_ids]
        elif plugin_ids==None:      
            plugin_ids = torch.arange(self.plugin_labels_num-1)
            conditions = conditions[plugin_ids]
        else:
            raise ValueError(f"plugin_ids has an unknown type: {type(plugin_ids)}")



        if self.cfg.datamodule.type=='MSD':
            audio = batch[0]
            audio = audio.squeeze(0)
            trackname = batch[2]
        else:
            audio = batch['waveform']
            if audio.dim()==3:
                audio = audio.squeeze(0)
            trackname = batch['file_name'][0]

        if audio.shape[1]==0:
            print(f"{trackname} is empty, skip transcribing")
            return None


        midi_events = {}
        output_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []

        # --- 1. Predict probabilities ---
#         print('--- 1. Predict probabilities ---')        
        num_instruments = len(conditions)

        for condition in conditions:
            idx = condition.argmax().item()
#             print(f"Predicting {self.IX_TO_NAME[condition.argmax().item()]} ({idx}/{num_instruments})", end='\r')
            _output_dict = predict_probabilities(self.network, audio.squeeze(0), condition, self.segment_samples, self.seg_batch_size)

            for key in ['reg_onset_output', 'frame_output']:
                output_dict[key].append(_output_dict[key]) # (timesteps, 88)    

        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = torch.cat(output_dict[key], axis=-1) # (timesteps, 88*num_conditions)


        midi_events = postprocess_probabilities_to_midi_events(output_dict, plugin_ids, self.IX_TO_NAME, self.classes_num, self.post_processor)
        # cause of memory leakage?

        with open(os.path.join(self.evaluation_output_path,f"{trackname}.pkl"), 'wb') as f:
            pickle.dump(midi_events, f)
        midi_path = os.path.join(self.evaluation_output_path,f"{trackname}.mid")
        write_midi_events_to_midi_file(midi_events, midi_path, self.instrument_type)
        
        if return_roll==True:
            return output_dict


    def configure_optimizers(self):
        r"""Configure optimizer."""

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]

    def log_images(self, tensors, conditions, key, idx, logger):       
        fig, axes = plt.subplots(2,2, figsize=(12,5), dpi=100)
        for ax, tensor, condition in zip(axes.flatten(), tensors, conditions):
            plugin_name = self.IX_TO_NAME[int(condition.argmax())]
            ax.imshow(tensor.cpu().detach().t(), aspect='auto', origin='lower', cmap='jet')
            ax.set_title(plugin_name)
        plt.tight_layout()
        logger.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=logger.current_epoch)  
        plt.close(fig)
    
#     def log_images(self, tensor, key):
#         fig, axes = plt.subplots(2,2)
#         for idx, ax in enumerate(axes.flatten()):
#             ax.imshow(tensor[idx].cpu().detach().t(), aspect='auto', origin='lower', cmap='jet')    
#         self.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)                     




class BaselineTranscription(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        lr_lambda,
        batch_data_preprocessor,
        cfg
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            network: nn.Module
            list_programs: list of list, e.g., [
                ['0', '16', '33', '48'],    # Slakh2100 dataset
                ['percussion'],             # Groove dataset
            ]
            loss_function: func
            learning_rate, float, e.g., 1e-3
            lr_lambda: func
        """
        super().__init__()
        # TODO: Try to add save_parameters
        self.network = network
        self.learning_rate = cfg.lr
        self.lr_lambda = lr_lambda
        self.classes_num = network.classes_num
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.NAME_TO_IX = cfg.MIDI_MAPPING.NAME_TO_IX
        self.instrument_type = cfg.MIDI_MAPPING.type
        
        self.seg_batch_size = cfg.transcription.evaluation.seg_batch_size
        self.segment_samples = cfg.segment_seconds*SAMPLE_RATE
#         self.evaluation_output_path = cfg.evaluation.output_path
        self.evaluation_output_path = os.path.join(os.getcwd(), 'MIDI_output')
        self.pkl_dir = cfg.datamodule.pkl_dir    
        os.makedirs(self.evaluation_output_path, exist_ok=True)
        
        self.frame_threshold = cfg.transcription.postprocessor.args.frame_threshold

        
        self.batch_data_preprocessor = batch_data_preprocessor
#         self.post_processor = OnsetFramePostProcessor(**cfg.transcription.postprocessor)
        self.post_processor = getattr(PostProcessor, cfg.transcription.postprocessor.type)(**cfg.transcription.postprocessor.args)
        

        

        # self.all_programs = list(itertools.chain(*list_programs))
        # E.g., [['0', '16', '33', '48'], ['percussion']] -> ['0', '16', '33', '48', 'percussion']

    def training_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self          
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: dict, e.g., {
                'waveform': (batch_size, segment_samples),
                '0_reg_onset_roll': (batch_size, frames_num, classes_num),
                'percussion_reg_onset_roll': (batch_size, frames_num, classes_num),
                ...
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # from IPython import embed; embed(using=False); os._exit(0)
        batch = self.batch_data_preprocessor(batch)
        # from IPython import embed; embed(using=False); os._exit(0)


        # Forward.
        target_roll = batch['target_dict']['frame_roll'] # (B, num_inst_classes, T, F)
        
        outputs = self.network(batch['waveforms'])
        loss = F.binary_cross_entropy(outputs['frame_output'], target_roll)
        
        logger.log('Transcription_Loss/Train', loss)        

        return loss
    
    def on_after_backward(self):
        pass
    
    def validation_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self             
        valid_metrics = {}      
        
        # Logging loss
        batch = self.batch_data_preprocessor(batch)        
        target_roll = batch['target_dict']['frame_roll'] # (B, num_inst_classes, T, F)
        outputs = self.network(batch['waveforms'])
        loss = F.binary_cross_entropy(target_roll, outputs['frame_output'])

        # Calculate loss.
#         if batch_idx<4:
#             if self.current_epoch==0:
# #                     self.log_images(outputs['spec'].squeeze(1), f'Valid/spectrogram')
#                 self.log_images(target_roll, batch['plugin_ids'], f'Valid/target_roll', batch_idx, logger)
#             self.log_images(outputs['frame_output'].squeeze(1), batch['plugin_ids'], f'Valid/frame_output', batch_idx, logger)
                
        valid_metrics['Transcription_Loss/Valid']=loss
           
        
        logger.log_dict(valid_metrics)
        
        return loss
        

    def log_images(self, tensors, plugin_idxs, key, idx, logger):
        # tensors = (B,num_inst_classes, T, F)
        # plugin_idxs = list variable lenght        
        
        # visualize only the first sample in the batch
        fig, axes = plt.subplots(2,2, figsize=(12,5), dpi=100)
        plugin_idxs = plugin_idxs[0]
        tensors = tensors[0]
        for ax, idx in zip(axes.flatten(), plugin_idxs):
            idx = idx.item()
            plugin_name = self.IX_TO_NAME[idx]
            ax.imshow(tensors[idx].cpu().detach().t(), aspect='auto', origin='lower', cmap='jet')
            ax.set_title(plugin_name)
        plt.tight_layout()
        logger.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=logger.current_epoch)  
        plt.close(fig)
    
#     def log_images(self, tensor, key):
#         fig, axes = plt.subplots(2,2)
#         for idx, ax in enumerate(axes.flatten()):
#             ax.imshow(tensor[idx].cpu().detach().t(), aspect='auto', origin='lower', cmap='jet')    
#         self.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)         
        
    def test_step(self, batch, batch_idx, jointist=None):
        plugin_ids = torch.where(batch['instruments'][0]==1)[0] 
        conditions = torch.zeros((len(plugin_ids), self.plugin_labels_num), device=plugin_ids.device)
#         conditions.zero_()
        conditions.scatter_(1, plugin_ids.view(-1,1), 1)    

        audio = batch['waveform']
        trackname = batch['hdf5_name'][0]

        midi_events = {}
        output_dict = {}
        framewise_dict = {}
        flat_output_dict = {}
        flattarget_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []
            flat_output_dict[key] = []
            flattarget_dict[key] = []

        # --- 1. Predict probabilities ---
#         print('--- 1. Predict probabilities ---')        
        num_instruments = len(conditions)

        
        _output_dict = predict_probabilities_baseline(self.network, audio.squeeze(0), self.segment_samples, self.seg_batch_size)

        batch = self.batch_data_preprocessor(batch)
        # calculating framewise score
        target_roll = batch['target_roll'] # # (1, num_inst_classes, T, F)
        target_roll = target_roll.squeeze(0) # remove batch dimension
        pred_roll = _output_dict['frame_output'] # (num_inst_classes, T, F)
        timesteps = min(target_roll.shape[1], pred_roll.shape[1])
        
        # start instrument-wise evaluation
        framewise_dict[trackname] = {}
        flat_framewise_dict = {}        
        flatroll_list = []        
        for idx in plugin_ids:
            if idx!=38: # only flatten rolls for non-drums
                frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(
                    target_roll[idx,:timesteps].cpu().flatten(),
                    pred_roll[idx,:timesteps].detach().cpu().flatten()>self.frame_threshold,
                    average='binary')

    #             print(f"F1 calculation done-------------")
                framewise_dict[trackname][self.IX_TO_NAME[idx.item()]] = {
                    'precision': frame_p,
                    'recall': frame_r,
                    'f1': frame_f1}
                for key in ['reg_onset_output', 'frame_output']:
                    output_dict[key].append(_output_dict[key][idx]) # (num_inst,timesteps, 88)               
            
            # Flattening different instruments into a single flat roll
                if 'flatroll' in locals():
                    flatroll = torch.logical_or(flatroll, pred_roll[idx,:timesteps]>self.frame_threshold)
                    flattarget_roll = torch.logical_or(flattarget_roll, target_roll[idx,:timesteps])                
                else:
                    flatroll = (pred_roll[idx,:timesteps]>self.frame_threshold)
                    flattarget_roll = target_roll[idx,:timesteps]
            
 

        # Preparing flat rolls
        flat_output_dict['frame_output'].append(flatroll)
        flat_output_dict['reg_onset_output'].append(flatroll)  
        flattarget_dict['frame_output'].append(flattarget_roll)
        flattarget_dict['reg_onset_output'].append(flattarget_roll)
        
        flat_frame_p, flat_frame_r, flat_frame_f1, _ = precision_recall_fscore_support(flattarget_roll[:timesteps].cpu().flatten(),
                                                                        flatroll[:timesteps].cpu().flatten(),
                                                                        average='binary')
        
        flat_framewise_dict[trackname]  = {
            'precision': flat_frame_p,
            'recall': flat_frame_r,
            'f1': flat_frame_f1}        
                
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = torch.cat(output_dict[key], axis=-1) # (num_inst,timesteps, 88*num_conditions)
            flat_output_dict[key] = torch.cat(flat_output_dict[key], axis=-1).float()
            flattarget_dict[key] = torch.cat(flattarget_dict[key], axis=-1).float()            
#         torch.save(output_dict, f'output_dict.pt')
#         --- 2. Postprocess probabilities to MIDI events ---
#         print('--- 2. Postprocess probabilities to MIDI events ---')
        midi_events = postprocess_probabilities_to_midi_events(output_dict, plugin_ids, self.IX_TO_NAME, self.classes_num, self.post_processor)
        flat_midi_events = postprocess_probabilities_to_midi_events(flat_output_dict, plugin_ids[:1], self.IX_TO_NAME, self.classes_num, self.post_processor)
        flat_midi_labels = postprocess_probabilities_to_midi_events(flattarget_dict, plugin_ids[:1], self.IX_TO_NAME, self.classes_num, self.post_processor)      
        pickle.dump(midi_events, open(os.path.join(self.evaluation_output_path,f"{trackname}.pkl"), 'wb'))
        pickle.dump(flat_midi_events, open(os.path.join(self.evaluation_output_path,f"{trackname}.flat_pkl"), 'wb'))
        pickle.dump(flat_midi_labels, open(os.path.join(self.evaluation_output_path,f"{trackname}.label_pkl"), 'wb'))        

        # --- 3. Write MIDI events to audio ---
#         print('--- 3. Write MIDI events to audio ---')

        midi_path = os.path.join(self.evaluation_output_path,f"{trackname}.mid")
        write_midi_events_to_midi_file(midi_events, midi_path, self.instrument_type)
        return framewise_dict, flat_framewise_dict

    def test_epoch_end(self, outputs, jointist=None):
        """
          outputs = [
                     {dict1}, {dict2},
                    {dict1}, {dict2},
                     ...
                     {dict1}, {dict2}
                    ]
        """
        
        framewise_dict = {'framewise': {},
                          'flat_framewise': {}}
        for idx, output_dict in enumerate(outputs):
            trackname = list(output_dict[0].keys())[0] # key here is the trackname
            framewise_dict['framewise'][trackname] = outputs[idx][0][trackname]
            framewise_dict['flat_framewise'][trackname] = outputs[idx][1][trackname]
    
        if jointist:
            logger=jointist
        else:
            logger=self                     
        pred_path = self.evaluation_output_path
        label_path = os.path.join(self.pkl_dir, 'test')
        
        notewise_dict = evaluate_F1(pred_path, label_path)
        flat_notewise_dict = evaluate_flat_F1(pred_path)    
        pickle.dump(notewise_dict, open("notewise_dict.pkl", 'wb')) # saving the notewise_dict as pickle file
        pickle.dump(flat_notewise_dict, open("flat_notewise_dict.pkl", 'wb')) # saving the notewise_dict as pickle file    
        
        # computing piecewise metrics
        piecewise_frame_f1 = np.mean(piecewise_evaluation(framewise_dict, 'framewise', 'f1'))
        piecewise_note_f1 = np.mean(piecewise_evaluation(notewise_dict, 'note', 'f1'))
        piecewise_note_w_off_f1 = np.mean(piecewise_evaluation(notewise_dict, 'note_w_off', 'f1'))        
        
        # computing instrumentwise metrics
        
        ## frame-wise metrics
        instrument_wise_precision, instrument_wise_recall, instrument_wise_F1 = calculate_intrumentwise_statistics(notewise_dict, 'note')
        instrument_wise_precision_woff, instrument_wise_recall_woff, instrument_wise_F1_woff = calculate_intrumentwise_statistics(notewise_dict, 'note_w_off')
        instrument_wise_precision_frame, instrument_wise_recall_frame, instrument_wise_F1_frame = calculate_intrumentwise_statistics(framewise_dict, 'framewise')        
        
        ## calculate note-wise metrics
        precision_mean, precision_std = calculate_mean_std(instrument_wise_precision)
        recall_mean, recall_std = calculate_mean_std(instrument_wise_recall)
        F1_mean, F1_std = calculate_mean_std(instrument_wise_F1)
        
        ## calculate note-wise-with-offset metrics        
        precision_mean_woff, precision_std_woff = calculate_mean_std(instrument_wise_precision_woff)
        recall_mean_woff, recall_std_woff = calculate_mean_std(instrument_wise_recall_woff)
        F1_mean_woff, F1_std_woff = calculate_mean_std(instrument_wise_F1_woff)
        
        
        # computing mean and std
        ## for instrumentwise:
        precision_mean_frame, precision_std_frame = calculate_mean_std(instrument_wise_precision_frame)
        recall_mean_frame, recall_std_frame = calculate_mean_std(instrument_wise_recall_frame)
        F1_mean_frame, F1_std_frame = calculate_mean_std(instrument_wise_F1_frame)        
        
        ## for flat:
#         torch.save(framewise_dict, 'framewise_dict.pt')
        flat_p_mean_frame, flat_r_mean_frame, flat_f1_mean_frame = get_flat_average(framewise_dict, 'flat_framewise')
        flat_p_mean_note, flat_r_mean_note, flat_f1_mean_note = get_flat_average(flat_notewise_dict, 'note')
        flat_p_mean_note_w_off, flat_r_mean_note_w_off, flat_f1_mean_note_w_off = get_flat_average(flat_notewise_dict, 'note_w_off')        
#         summary = [[scores['precision'], scores['recall'], scores['f1']] for trackname, scores in framewise_dict['flat_framewise'].items()]
#         flat_p_mean_frame, flat_r_mean_frame, flat_f1_mean_frame = np.mean(summary, axis=0)  
        
        global_mean, fig_notef1 = barplot(F1_mean, 'Notewise F1 scores', figsize=(4,36))
        global_mean_woff, fig_noteoffsetf1 = barplot(F1_mean_woff, 'Notewise-offset F1 scores', figsize=(4,36))
        global_mean_frame, fig_frame = barplot(F1_mean_frame, 'Framewise F1 scores', figsize=(4,36))        
        
        logger.logger.experiment.add_figure(f"Test/Notewise F1 scores",
                                            fig_notef1,
                                            global_step=self.current_epoch)
        logger.logger.experiment.add_figure(f"Test/Notewise offset F1 scores",
                                            fig_noteoffsetf1,
                                            global_step=self.current_epoch)
        logger.logger.experiment.add_figure(f"Test/Framewise F1 scores",
                                            fig_frame,
                                            global_step=self.current_epoch)
        
        logger.log(f"Test/notewise", global_mean, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/notewise_w_offset", global_mean_woff, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Framewise", global_mean_frame, on_step=False, on_epoch=True, rank_zero_only=True)
                                   
        logger.log(f"Test/piecewise_notewise", piecewise_note_f1, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/piecewise_notewise_w_offset", piecewise_note_w_off_f1, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/piecewise_Framewise", piecewise_frame_f1, on_step=False, on_epoch=True, rank_zero_only=True)                                   
        
        
        logger.log(f"Test/Flat_Framewise", flat_f1_mean_frame, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Flat_notewise", flat_f1_mean_note, on_step=False, on_epoch=True, rank_zero_only=True)
        logger.log(f"Test/Flat_notewise_w_offset", flat_f1_mean_note_w_off, on_step=False, on_epoch=True, rank_zero_only=True)      
        
    def predict_step(self, batch, batch_idx):
        plugin_ids = torch.where(batch['instruments'][0]==1)[0] 
        conditions = torch.zeros((len(plugin_ids), self.plugin_labels_num), device=plugin_ids.device)
#         conditions.zero_()
        conditions.scatter_(1, plugin_ids.view(-1,1), 1)    

        audio = batch['waveform']
        trackname = batch['hdf5_name'][0]

        midi_events = {}
        output_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []

        # --- 1. Predict probabilities ---
#         print('--- 1. Predict probabilities ---')        
        num_instruments = len(conditions)
        for condition in conditions:
            idx = condition.argmax().item()
#             print(f"Predicting {self.IX_TO_NAME[condition.argmax().item()]} ({idx}/{num_instruments})", end='\r')
            _output_dict = predict_probabilities(self.network, audio.squeeze(0), condition, self.segment_samples, self.seg_batch_size)

            for key in ['reg_onset_output', 'frame_output']:
                output_dict[key].append(_output_dict[key]) # (timesteps, 88)    

        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = torch.cat(output_dict[key], axis=-1) # (timesteps, 88*num_conditions)   
            
        return output_dict


    def configure_optimizers(self):
        r"""Configure optimizer."""

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]