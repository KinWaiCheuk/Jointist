from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor, FullPreprocessor
from End2End.tasks.separation import Separation
import End2End.models.separation as SeparationModel

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
import End2End.losses as Losses

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path




@hydra.main(config_path="End2End/config/", config_name="separation_config")
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

    if cfg.MIDI_MAPPING.type=='plugin_names':
        cfg.MIDI_MAPPING.plugin_labels_num = PLUGIN_LABELS_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = PLUGIN_LB_TO_IX
        cfg.MIDI_MAPPING.IX_TO_NAME = PLUGIN_IX_TO_LB
        cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes3/')   
    elif cfg.MIDI_MAPPING.type=='MIDI_class':
        cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
        cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_class/')
    else:
        raise ValueError(f"Please choose the correct MIDI_MAPPING.type")        

    Model = getattr(SeparationModel, cfg.separation.model.type)
    if cfg.separation.model.type=='CondUNet':
        model = Model(**cfg.separation.model.args)
        cfg.transcription = False
    elif cfg.separation.model.type=='TCondUNet':
        model = Model(**cfg.separation.model.args, spec_cfg=cfg.feature)
        cfg.transcription = True        
    else:
        raise ValueError("please choose the correct model type")

    
    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None

    # data module
    data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=augmentor, MIDI_MAPPING=cfg.MIDI_MAPPING)
    data_module.setup()

    experiment_name = (
                      f"{cfg.separation.model.type}-"    
                      f"{cfg.inst_sampler.samples}p{cfg.inst_sampler.neg_samples}n-"
                      f"csize={MIDI_Class_NUM}"
                      )
    DataPreprocessor = End2EndBatchDataPreprocessor
    # loss function
    loss_function = getattr(Losses, cfg.separation.model.loss_types)

    # callbacks
    # save checkpoint callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')    
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint,
                                          auto_insert_metric_name=False)
    callbacks = [checkpoint_callback, lr_monitor]
    
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)

    # learning rate reduce function.
    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)


    pl_model = Separation(
        network=model,
        loss_function=loss_function,
        lr_lambda=lr_lambda,
        batch_data_preprocessor=DataPreprocessor(**cfg.separation.batchprocess),
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
    trainer.fit(pl_model, data_module)
    trainer.test(pl_model, data_module.test_dataloader())


if __name__ == '__main__':
    main()
