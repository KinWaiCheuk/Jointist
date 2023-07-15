import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

# from End2End.inference_instruments_filter import RegressionPostProcessor, OnsetFramePostProcessor
import End2End.inference_instruments_filter as PostProcessor
from End2End.constants import SAMPLE_RATE, FRAMES_PER_SECOND

import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from End2End.tasks.separation.utils import calculate_sdr, _append_to_dict, barplot
import torchaudio



from pathlib import Path

class Separation(pl.LightningModule):
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
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.instrument_type = cfg.MIDI_MAPPING.type
        
        self.segment_samples = cfg.segment_seconds*SAMPLE_RATE
        self.seg_batch_size = cfg.separation.evaluation.seg_batch_size        
#         self.evaluation_output_path = cfg.evaluation.output_path
        self.evaluation_output_path = os.path.join(os.getcwd(), 'audio_output')
        self.notes_pkls_dir = cfg.datamodule.notes_pkls_dir    
        os.makedirs(self.evaluation_output_path, exist_ok=True)
        
        self.transcription = cfg.transcription
        
        if hasattr(cfg.datamodule, 'dataset_cfg'):
            self.test_segment_size = cfg.datamodule.dataset_cfg.test.segment_seconds
        self.batch_data_preprocessor = batch_data_preprocessor
        

        

        # self.all_programs = list(itertools.chain(*list_programs))
        # E.g., [['0', '16', '33', '48'], ['percussion']] -> ['0', '16', '33', '48', 'percussion']

    def training_step(self, batch, batch_idx, roll_dict=None, jointist=None):
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
        if self.transcription:
            if roll_dict!=None:
                frame_roll = roll_dict['frame_output']
            else:
                roll_dict = batch['target_dict']
                frame_roll = roll_dict['frame_roll']
            # keys = {frame_roll, onset_roll}
            outputs = self.network(batch['waveforms'].unsqueeze(1), batch['conditions'], frame_roll)
        else:
            outputs = self.network(batch['waveforms'].unsqueeze(1), batch['conditions'])
        # outputs['waveform'].shape (B, 1, len)
        valid_B = int(batch['source_masks'].sum().item())
        audio_len = batch['sources'].shape[-1]    
        
        masked_labels = torch.masked_select(batch['sources'], batch['source_masks'].bool()).view(valid_B, audio_len)        
        masked_outputs = torch.masked_select(outputs['waveform'].squeeze(1), batch['source_masks'].bool()).view(valid_B, audio_len)        

            # from IPython import embed; embed(using=False); os._exit(0)               
        loss = self.loss_function(masked_outputs, masked_labels) # self.network is not needed        
        
        logger.log('Separation/Train/Loss', loss)
        mean_sdr = torch.mean(calculate_sdr(masked_labels, masked_outputs))
        logger.log('Separation/Train/SDR', mean_sdr)

        return loss
    
    def validation_step(self, batch, batch_idx, roll_dict=None, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self             
        valid_metrics = {}      
        
        # Logging loss
        if self.batch_data_preprocessor:
            batch = self.batch_data_preprocessor(batch)        
        # batch['waveforms'].unsqueeze(1) (B, len) -> (B, 1, len)
        if self.transcription:
            if roll_dict!=None:
                frame_roll = roll_dict['frame_output']
            else:
                roll_dict = batch['target_dict']
                frame_roll = roll_dict['frame_roll']
            # keys = {frame_roll, onset_roll}
            outputs = self.network(batch['waveforms'].unsqueeze(1), batch['conditions'], frame_roll)
        else:
            outputs = self.network(batch['waveforms'].unsqueeze(1), batch['conditions'])
            
        valid_B = int(batch['source_masks'].sum().item())
        audio_len = batch['sources'].shape[-1]                  
        
        masked_labels = torch.masked_select(batch['sources'], batch['source_masks'].bool()).view(valid_B, audio_len)        
        masked_outputs = torch.masked_select(outputs['waveform'].squeeze(1), batch['source_masks'].bool()).view(valid_B, audio_len)       
        
        
        if batch_idx==0:
            if self.current_epoch==0:
                logger.logger.experiment.add_audio(f'Mix/0', batch['waveforms'][0], sample_rate=16000)            
            for idx, condition in enumerate(batch['conditions']):
                name = self.IX_TO_NAME[torch.argmax(condition).cpu().item()]
                logger.logger.experiment.add_audio(f'Labels/{name}', batch['sources'][idx], sample_rate=16000, global_step=self.current_epoch)
                logger.logger.experiment.add_audio(f'Preds/{name}', outputs['waveform'][idx], sample_rate=16000, global_step=self.current_epoch)
    
        loss = self.loss_function(masked_outputs, masked_labels) # self.network is not needed
        valid_metrics['Separation/Valid/Loss']=loss
        
        mean_sdr = torch.mean(calculate_sdr(masked_labels, masked_outputs))
        valid_metrics['Separation/Valid/SDR']=mean_sdr
        logger.log_dict(valid_metrics)
        
        return valid_metrics
        
    def test_step(self, batch, batch_idx, roll_pred=None, jointist=None):
        
        # batch['waveform'] = (len)
        if jointist:
            logger=jointist
        else:
            logger=self             
        valid_metrics = {}      
        
        # getting conditions
        # When testing segments roll_pred['frame_output']=(1, T, F)
        # When full lenght roll_pred['frame_output']=(T, F)        
        
        # debugging padding problems
#         print(f"{batch['waveform'].shape=}")
#         print(f"{batch['target_dict'][0]['Piano']['frame_roll'].shape=}")
#         print(f"{roll_pred['frame_output'].shape=}")
        
        plugin_ids = torch.where(batch['instruments'][0]==1)[0] 
        conditions = torch.zeros((len(plugin_ids), self.plugin_labels_num), device=plugin_ids.device)
#         conditions.zero_()
        conditions.scatter_(1, plugin_ids.view(-1,1), 1)    
        trackname = batch['hdf5_name'][0]
        if batch_idx==0:
            Path(os.path.join(self.evaluation_output_path, trackname)).mkdir(parents=True, exist_ok=False)        
        # batch['waveforms'].unsqueeze(1) (B, len) -> (B, 1, len)
        sdr_dict = {}
        sdr_dict[trackname] = {}
        for idx, condition in enumerate(conditions):
            if self.test_segment_size!=None:
                # TODO: use a better condition rather than this dangerous one
                # since self.transcirption is always=True
                if roll_pred!=None:
                    roll = roll_pred['frame_output'].squeeze(0)[:,idx*88:(idx+1)*88]
                    # keys = {'Strings', 'Piano', ... 'Bass'}
                    roll = roll.float().to(condition.device).unsqueeze(0) # (1, T, F)

                    roll_dict = batch['target_dict'][0]
                    target_roll = torch.from_numpy(roll_dict[self.IX_TO_NAME[condition.argmax().item()]]['frame_roll']) # (T, F)
                    target_roll = target_roll.float().unsqueeze(0).to(condition.device) # (1, T, F)                       
#                     timesteps = min(target_roll.shape[1], roll.shape[1])
                    
#                     frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(roll[0,:timesteps].cpu().flatten(),
#                                                                                     target_roll[0,:timesteps].cpu().flatten(),
#                                                                                     average='binary')     
#                     print(f"{frame_f1=}")
                    _output_dict = self.network(batch['waveform'].unsqueeze(1), condition.unsqueeze(0), roll.float()) 
                elif self.transcription:
                    roll_dict = batch['target_dict'][0]
                    # keys = {'Strings', 'Piano', ... 'Bass'}
                    roll = torch.from_numpy(roll_dict[self.IX_TO_NAME[condition.argmax().item()]]['frame_roll']) # (T, F)
                    roll = roll.float().unsqueeze(0).to(condition.device) # (1, T, F)                    
                    _output_dict = self.network(batch['waveform'].unsqueeze(1), condition.unsqueeze(0), roll.float())                   
                else:
                    _output_dict = self.network(batch['waveform'].unsqueeze(1), condition.unsqueeze(0))
            else:
                if roll_pred!=None:
                    roll = roll_pred['frame_output'][:,idx*88:(idx+1)*88]
                    # keys = {'Strings', 'Piano', ... 'Bass'}
                    roll = roll.float().to(condition.device) # (1, T, F)
                    roll.unsqueeze(0)                                  
                    _output_dict = self._separate_fullaudio(batch['waveform'], condition, self.segment_samples, self.seg_batch_size, roll)         
                elif self.transcription:
                    roll_dict = batch['target_dict'][0]
                    # keys = {'Strings', 'Piano', ... 'Bass'}
                    roll = torch.from_numpy(roll_dict[self.IX_TO_NAME[condition.argmax().item()]]['frame_roll']) # (T, F)
                    roll = roll.float().to(condition.device) # (1, T, F)
                    _output_dict = self._separate_fullaudio(batch['waveform'], condition, self.segment_samples, self.seg_batch_size, roll)                  
                else:
                    _output_dict = self._separate_fullaudio(batch['waveform'], condition, self.segment_samples, self.seg_batch_size)            
            pred = _output_dict['waveform'] # (len)
        
            plugin_type = self.IX_TO_NAME[plugin_ids[idx].item()]
            
            # adjust the output dim
            if pred.dim()==1:
                pred = pred.unsqueeze(0)
            elif pred.dim()==3:
                pred = pred.squeeze(1)
                
            if batch_idx==0:
                torchaudio.save(os.path.join(self.evaluation_output_path, batch['hdf5_name'][0],f"{plugin_type}.mp3"),
                                pred.cpu(),
                                16000)
                torch.save(_output_dict, os.path.join(self.evaluation_output_path, batch['hdf5_name'][0],f"{plugin_type}.pt"))
            # calcuating and logging SDR
            # 1. convert numpy array into torch tensor
            label = torch.from_numpy(batch['sources'][0][plugin_type])
            # 2. make sure the source shape is (1, len)
            if label.dim()==1:
                label = label.unsqueeze(0)
            elif label.dim()==2:
                label = label
            else:
                raise ValueError(f"label shape = {label.shape}. Please make sure it is (1, len)")

            sdr = calculate_sdr(label, pred.cpu())
            logger.log('Separation/Test/sourcewise_SDR', sdr)
            sdr_dict[trackname][plugin_type] = sdr
        
        
        return sdr_dict

    def test_epoch_end(self, outputs, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self             
        
        piecewise_sdr = {}
        instrumentwise_sdr = {}
        for track in outputs:
            trackname = list(track.keys())[0] # get the trackname
            piecewise_sdr[trackname] = (sum(track[trackname].values())/len(track[trackname]))

            # appending instrument-wise sdr
            for inst, sdr in track[trackname].items():
                _append_to_dict(instrumentwise_sdr, inst, sdr.item())

        # averaging each instrument
        for key, items in instrumentwise_sdr.items():
            instrumentwise_sdr[key] = np.mean(items)
            
        mean_instrumentwise_sdr, _ = barplot(instrumentwise_sdr, f"inst_wise-test_len_{self.test_segment_size}")      
        mean_piecewise_sdr, _ = barplot(piecewise_sdr, f"piece_wise-test_len {self.test_segment_size}", figsize=(4,40))
        
        torch.save(instrumentwise_sdr, f"instrumentwise_sdr.pt")
        torch.save(piecewise_sdr, f"piecewise_sdr.pt")        
        
        logger.log('Separation/Test/mean_instrumentwise_sdr', mean_instrumentwise_sdr)
        logger.log('Separation/Test/mean_piecewise_sdr', mean_piecewise_sdr)
        
        
    def predict_step(self, batch, batch_idx, roll_pred, plugin_ids):
        
        conditions = torch.eye(self.plugin_labels_num).to(batch['waveform'].device) # assume all instruments are presented
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
        
        audio = batch['waveform']
        if audio.dim()==3:
            audio = audio.squeeze(0)
        trackname = batch['file_name'][0]
        
        if audio.shape[1]==0:
            print(f"{trackname} is empty, skip transcribing")
            return None        
        
        Path(os.path.join(self.evaluation_output_path, trackname)).mkdir(parents=True, exist_ok=False)        
        # batch['waveforms'].unsqueeze(1) (B, len) -> (B, 1, len)

        for idx, condition in enumerate(conditions):
            roll = roll_pred['frame_output'][:,idx*88:(idx+1)*88]
            # keys = {'Strings', 'Piano', ... 'Bass'}
            roll = roll.float().to(condition.device) # (1, T, F)
            roll.unsqueeze(0)                                  
            _output_dict = self._separate_fullaudio(batch['waveform'], condition, self.segment_samples, self.seg_batch_size, roll)                  
            pred = _output_dict['waveform'] # (len)
        
            plugin_type = self.IX_TO_NAME[plugin_ids[idx].item()]
            
            # adjust the output dim
            if pred.dim()==1:
                pred = pred.unsqueeze(0)
            elif pred.dim()==3:
                pred = pred.squeeze(1)
                
            torchaudio.save(os.path.join(self.evaluation_output_path, trackname,f"{plugin_type}.mp3"),
                            pred.cpu(),
                            16000)


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


    def _separate_fullaudio(self, audio, condition, segment_samples, segment_batch_size, roll=None, roll_segment_samples=1001):
        # TODO, refactor segment_samples, model, segment_batch_size
        # roll_segment_sample is hard coded to 1001 (hop_length=160, sr=16000)

        r"""Separate full audio by cutting them into small segments

        Args:
            audio: (audio_samples,).

        Returns:
            ???
        """
#         audio = audio[None, :]  # (1, audio_samples) 

        # Pad audio to be evenly divided by segment_samples.
        audio_length = audio.shape[1]
#         pad_len = int(np.ceil(audio_length / segment_samples)) * segment_samples - audio_length

        # I don't know why it works, but it works.
        # No time to check why is the padding like this        
        pad_len = segment_samples - (audio_length-segment_samples)%(segment_samples//2)         

    #     audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
        audio = torch.cat((audio, torch.zeros((1, pad_len), device=audio.device)), axis=1)

        # Enframe to segments.
    #     segments = enframe(audio, segment_samples)
        segments = audio.unfold(1, segment_samples, segment_samples//2).squeeze(0) # faster version of enframe
        # (N, segment_samples)
        
    #     conditions = np.tile(condition, (len(segments), 1))
        conditions = condition.unsqueeze(0).repeat(len(segments),1).to(audio.device)
        
        
        
        if roll!=None:
            roll = roll[None, :]
            roll_length = roll.shape[1]
            bins = roll.shape[2]
            
#             roll_pad_len = int(np.ceil(roll_length / roll_segment_samples)) * roll_segment_samples - roll_length
            # I don't know why it works, but it works.
            # No time to check why is the padding like this
            roll_pad_len = roll_segment_samples - (roll_length-roll_segment_samples)%(roll_segment_samples//2) 
            
            # pad according to waveform samples
            # roll_pad_len = int(pad_len/160)+1 # It causes problems
            roll = torch.cat((roll, torch.zeros((1, roll_pad_len, bins), device=audio.device)), axis=1)
            # roll (1, T, F)
            roll_segments = roll.unfold(1, roll_segment_samples, roll_segment_samples//2).squeeze(0) # faster version of enframe
            # roll_segments (N, F, T)
            roll_segments = roll_segments.transpose(-1,-2)
            # Inference on segments.
        #     output_dict = _forward_mini_batches(model, segments, conditions, batch_size=batch_size)
            output_dict = self._forward_mini_batches_torch(segments, conditions, roll_segments, batch_size=segment_batch_size)
            # {'waveform': (len)}        
        
        else:
            # Inference on segments.
        #     output_dict = _forward_mini_batches(model, segments, conditions, batch_size=batch_size)
            output_dict = self._forward_mini_batches_torch(segments, conditions, batch_size=segment_batch_size)
            # {'waveform': (len)}


        # Deframe to original length.
        for key in output_dict.keys():
            if key=='waveform':
                X = output_dict[key].squeeze(1)
                output_dict[key] = torch.cat((X[0,:120000], X[1:-1,40000:120000].flatten(0,1), X[-1,40000:]),0)[:audio_length].cpu()  # faster version of deframe                         
            elif key=='roll_feat':
                audio_duration = audio_length / SAMPLE_RATE
                frames_num = int(audio_duration * FRAMES_PER_SECOND)                
                X = output_dict[key].squeeze(1)[:,:-1]
                print(f"{X.shape=}")
                output_dict[key] = torch.cat((X[0,:750], X[1:-1,250:750].flatten(0,1), X[-1,250:]),0)[:frames_num].cpu()  # faster version of deframe                

        return output_dict
    
    
    def _forward_mini_batches_torch(self, x, conditions, roll_segments=None, batch_size=8):
        r"""Forward data to model in mini-batch.

        Args:
            model: nn.Module
            x: ndarray, (N, segment_samples)
            batch_size: int

        Returns:
            output_dict: dict, e.g. {
                'frame_output': (segments_num, frames_num, classes_num),
                'onset_output': (segments_num, frames_num, classes_num),
                ...}
        """
        output_dict = {}
        
        # quick patch to solve segment number mismatch
        if roll_segments!=None:
            min_segs =  min(x.shape[0], roll_segments.shape[0])
            x = x[:min_segs]
            conditions = conditions[:min_segs]
            roll_segments = roll_segments[:min_segs]

        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = x[pointer : pointer + batch_size]
            batch_condition = conditions[pointer : pointer + batch_size]

            if roll_segments!=None:
                roll_segment = roll_segments[pointer:pointer+batch_size]
                # batch_waveform (B, len)
                batch_output_dict = self.network(batch_waveform.unsqueeze(1), batch_condition, roll_segment)
            else:
                # batch_waveform (B, len)
                batch_output_dict = self.network(batch_waveform.unsqueeze(1), batch_condition)
            
            pointer += batch_size
            for key in batch_output_dict.keys():
                _append_to_dict(output_dict, key, batch_output_dict[key])

        for key in output_dict.keys():
            output_dict[key] = torch.cat(output_dict[key], axis=0)

        return output_dict