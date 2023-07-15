from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor, FullPreprocessor
from End2End.tasks.transcription import Transcription, BaselineTranscription
from End2End.models.transcription.seg_baseline import Semantic_Segmentation

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
import End2End.models.transcription.combined as TranscriptionModel
from End2End.losses import get_loss_function

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path




@hydra.main(config_path="End2End/config/", config_name="transcription_config")
def main(cfg):
    r"""Train an instrument classification system, evluate, and save checkpoints.

    Args:
        workspace: str, path
        config_yaml: str, path
        gpus: int
        mini_data: bool

    Returns:
        None
    """
    
    cfg.datamodule.waveform_dir = to_absolute_path(cfg.datamodule.waveform_dir)
    cfg.datamodule.slakhdata_root =  to_absolute_path(cfg.datamodule.slakhdata_root)
    cfg.datamodule.pkl_dir = to_absolute_path(cfg.datamodule.pkl_dir) 

    if cfg.MIDI_MAPPING.type=='plugin_names':
        cfg.MIDI_MAPPING.plugin_labels_num = PLUGIN_LABELS_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = PLUGIN_LB_TO_IX
        cfg.MIDI_MAPPING.IX_TO_NAME = PLUGIN_IX_TO_LB
    elif cfg.MIDI_MAPPING.type=='MIDI_class':
        cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
    else:
        raise ValueError(f"Please choose the correct MIDI_MAPPING.type")        

    

    
    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None

    # data module
    data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=augmentor, MIDI_MAPPING=cfg.MIDI_MAPPING)
    data_module.setup('test')

    # model
    if cfg.transcription.model.type=='Semantic_Segmentation':
        model = Semantic_Segmentation(cfg, **cfg.transcription.model.args)
        experiment_name = (
                          f"Eval-"
                          f"{cfg.transcription.model.type}-"
                          f"{cfg.MIDI_MAPPING.type}-"
                          f"csize={MIDI_Class_NUM}-"
                          f"bz={cfg.batch_size}"
                          )
        DataPreprocessor = FullPreprocessor
        # loss function       
    else:
        Model = getattr(TranscriptionModel, cfg.transcription.model.type)
        model = Model(cfg, **cfg.transcription.model.args)
        experiment_name = (
                          f"Eval-"            
                          f"{cfg.transcription.model.type}-"
                          f"{cfg.transcription.backend.acoustic.type}-"
                          f"{cfg.transcription.backend.language.type}_"
                          f"{cfg.transcription.backend.language.args.hidden_size}-"        
                          f"{cfg.MIDI_MAPPING.type}-"
                          f"{cfg.inst_sampler.mode}_{cfg.inst_sampler.temp}_{cfg.inst_sampler.samples}inst_"
                          f"noise{cfg.inst_sampler.audio_noise}-"
                          f"fps={cfg.transcription.model.args.frames_per_second}-"
                          f"csize={MIDI_Class_NUM}-"
                          f"bz={cfg.batch_size}"
                          )
        # loss function
        loss_function = get_loss_function(cfg.transcription.model.loss_types)

    # callbacks
    # save checkpoint callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')    
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint,
                                          auto_insert_metric_name=False)
    callbacks = [checkpoint_callback, lr_monitor]
    
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)
    
    checkpoint_path = to_absolute_path(cfg.transcription.evaluation.checkpoint_path)    
    
    ckpt = torch.load(checkpoint_path)
    new_state_dict = {}
    for key in ckpt['state_dict'].keys():
        if 'transcription_model' in key:
            new_key = '.'.join(key.split('.')[2:])
            new_state_dict[new_key] = ckpt['state_dict'][key]    
        else:
            if 'network.'  in key:
                 new_key = '.'.join(key.split('.')[1:])
                 new_state_dict[new_key] = ckpt['state_dict'][key]    
#         else:
#             raise ValueError(f'Unexpected key {key}')
    
    model.load_state_dict(new_state_dict)    

    if cfg.transcription.model.type=='Semantic_Segmentation':
        # PL model       
        pl_model = BaselineTranscription.load_from_checkpoint(checkpoint_path,
                                                      network=model,
                                                      lr_lambda=None,
                                                      batch_data_preprocessor=End2EndBatchDataPreprocessor(
                                                          **cfg.transcription.batchprocess),
                                                      cfg=cfg)        
    else:        
        # PL model
        pl_model = Transcription(
            network=model,
            loss_function=None,
            lr_lambda=None,
            batch_data_preprocessor=End2EndBatchDataPreprocessor(
                **cfg.transcription.batchprocess,
                transcription=True,
                source_separation=False),
            cfg=cfg        
        )
  

    if cfg.trainer.gpus==0: # If CPU is used, disable syncbatch norm
        cfg.trainer.sync_batchnorm=False    
    
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=[DDPPlugin(find_unused_parameters=False)],
        logger=logger  
    )
    
    

    # Fit, evaluate, and save checkpoints.
    trainer.test(pl_model, data_module.test_dataloader())


if __name__ == '__main__':
    main()

