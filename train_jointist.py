from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor
from End2End.tasks import Jointist, Transcription, Detection
import End2End.models.instrument_detection as DectectionModel
from End2End.models.transcription.instruments_filter_models import get_model_class

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
from End2End.losses import get_loss_function

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path




@hydra.main(config_path="End2End/config/", config_name="Jointist")
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

    cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
    cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
    cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
    cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_class/')         
    
    experiment_name = "jointist".format(cfg.gpus, cfg.batch_size, cfg.segment_seconds)

    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None

    # data module
    data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=augmentor, MIDI_MAPPING=cfg.MIDI_MAPPING)

    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)    
    # defining transcription model
    Model = get_model_class(cfg.transcription.model.type)
    model = Model(cfg.feature, **cfg.transcription.model.args)
    loss_function = get_loss_function(cfg.transcription.model.loss_types)
    transcription_model = Transcription(
        network=model,
        loss_function=loss_function,
        lr_lambda=lr_lambda,
        batch_data_preprocessor=End2EndBatchDataPreprocessor(cfg.MIDI_MAPPING, 'random'),
        cfg=cfg
    )    
    

    # defining instrument detection model
    Model = getattr(DectectionModel, cfg.detection.model.type)
    model = Model(num_classes=cfg.MIDI_MAPPING.plugin_labels_num, spec_args=cfg.feature, **cfg.detection.model.args)
    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)
    detection_model = Detection(
        network=model,
        lr_lambda=lr_lambda,
        cfg=cfg
    )
    
    # defining jointist
    jointist = Jointist(
        detection_model=detection_model,
        transcription_model=transcription_model,
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
        plugins=[DDPPlugin(find_unused_parameters=True)],
        logger=logger        
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(jointist, data_module)


if __name__ == '__main__':
    main()
