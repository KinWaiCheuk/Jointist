from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor
from End2End.tasks.transcription import Transcription

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
    cfg.datamodule.waveform_hdf5s_dir = to_absolute_path(os.path.join('hdf5s', 'waveforms'))
    if cfg.transcription.evaluation.output_path:
        cfg.transcription.evaluation.output_path = to_absolute_path(cfg.transcription.evaluation.output_path)
    else:
        cfg.transcription.evaluation.output_path = os.getcwd()
    
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
        
    experiment_name = ("Eval-"
                      f"{cfg.transcription.model.type}-"
                      f"{cfg.MIDI_MAPPING.type}-"
                      f"hidden=128-"
                      f"fps={cfg.transcription.model.args.frames_per_second}-"
                      f"csize={MIDI_Class_NUM}-"
                      f"bz={cfg.batch_size}"
                      )
        
    
    data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=None, MIDI_MAPPING=cfg.MIDI_MAPPING)
    data_module.setup('test')
    
    checkpoint_path = to_absolute_path(cfg.transcription.evaluation.checkpoint_path)

    # model
    Model = getattr(TranscriptionModel, cfg.transcription.model.type)
    model = Model(cfg, **cfg.transcription.model.args)  
    
    pl_model = Transcription.load_from_checkpoint(checkpoint_path,
                                                network=model,
                                                loss_function=None,
                                                lr_lambda=None,
                                                batch_data_preprocessor=End2EndBatchDataPreprocessor(cfg.MIDI_MAPPING, cfg.inst_sampler.type, cfg.inst_sampler.temp),
                                                cfg=cfg)

    
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)
    trainer = pl.Trainer(
        **cfg.trainer,
        plugins=[DDPPlugin(find_unused_parameters=True)],
        logger=logger
    )
    trainer.test(pl_model, data_module.test_dataloader())
    
if __name__ == '__main__':
    main()