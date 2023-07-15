from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor
from End2End.tasks.separation import Separation
from End2End.tasks.transcription import Transcription
from End2End.tasks.t_separation import TSeparation

import End2End.models.separation as SeparationModel
import End2End.models.transcription.combined as TranscriptionModel

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
from End2End.losses import get_loss_function
import End2End.losses as Losses

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path




@hydra.main(config_path="End2End/config/", config_name="tseparation")
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
    
    
    cfg.datamodule.waveform_hdf5s_dir = to_absolute_path(os.path.join('hdf5s', 'waveforms'))
    if cfg.trainer.resume_from_checkpoint: # resume previous training when this is given
        cfg.trainer.resume_from_checkpoint = to_absolute_path(cfg.trainer.resume_from_checkpoint)   

    cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
    cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
    cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
    cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_class/')         
    
    if cfg.transcription_weights!=None:
        experiment_name = (
                          f"TSeparation-{cfg.inst_sampler.samples}p{cfg.inst_sampler.neg_samples}n-"
                          f"ste_roll-pretrainedT"
                          )        
    else:
        experiment_name = (
                          f"TSeparation-{cfg.inst_sampler.samples}p{cfg.inst_sampler.neg_samples}n-"
                          f"ste_roll"
                          )

    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None

    # data module
    data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=augmentor, MIDI_MAPPING=cfg.MIDI_MAPPING)
    data_module.setup()

    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)
    
    
    # loss function
    loss_function = getattr(Losses, cfg.separation.model.loss_types)
    model = getattr(SeparationModel, cfg.separation.model.type)\
                              (**cfg.separation.model.args, spec_cfg=cfg.separation.feature)
    
    separation_model = Separation(
        network=model,
        loss_function=loss_function,
        lr_lambda=lr_lambda,
        batch_data_preprocessor=None,
        cfg=cfg
    )        
    
    # defining transcription model
    Model = getattr(TranscriptionModel, cfg.transcription.model.type)
    model = Model(cfg, **cfg.transcription.model.args)
    loss_function = get_loss_function(cfg.transcription.model.loss_types)
    
    if cfg.transcription_weights!=None:    
        checkpoint_path = to_absolute_path(cfg.transcription_weights)
        transcription_model = Transcription.load_from_checkpoint(checkpoint_path,
                                                    network=model,
                                                    loss_function=loss_function,
                                                    lr_lambda=lr_lambda,
                                                    batch_data_preprocessor=None,
                                                    cfg=cfg)
    else:
        transcription_model = Transcription(
            network=model,
            loss_function=loss_function,
            lr_lambda=lr_lambda,
            batch_data_preprocessor=None,
            cfg=cfg
        )    
    
    
    # defining jointist
    tseparation = TSeparation(
        transcription_model = transcription_model,
        separation_model = separation_model,
        batch_data_preprocessor = End2EndBatchDataPreprocessor(cfg.MIDI_MAPPING,
                                                             **cfg.inst_sampler,
                                                             transcription=True,
                                                             source_separation=True),        
        lr_lambda=lr_lambda,
        cfg=cfg
    )    
    

    # defining Trainer
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint, auto_insert_metric_name=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor]
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=[DDPPlugin(find_unused_parameters=False)],
        logger=logger,        
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(tseparation, data_module)
    trainer.test(tseparation, data_module)

if __name__ == '__main__':
    main()
