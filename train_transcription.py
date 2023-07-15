from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor, FullPreprocessor
from End2End.tasks.transcription import Transcription, BaselineTranscription
from End2End.models.transcription.seg_baseline import Semantic_Segmentation

# from End2End.MIDI_program_map import (
#                                       MIDI_Class_NUM,
#                                       MIDIClassName2class_idx,
#                                       class_idx2MIDIClass,
#                                       )

from slakh_loader.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
import End2End.models.transcription.combined as TranscriptionModel
from End2End.losses import get_loss_function
from slakh_loader.slakh2100 import SlakhCollator

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
    data_module.setup()

    # model
    if cfg.transcription.model.type=='Semantic_Segmentation':
        cfg.transcription.model.args.out_class = cfg.transcription.model.args.out_class-1 # remove empty class
        model = Semantic_Segmentation(cfg, **cfg.transcription.model.args)
        experiment_name = (
                          f"{cfg.transcription.model.type}-"
                          f"{cfg.MIDI_MAPPING.type}-"
                          f"csize={MIDI_Class_NUM}-"
                          f"bz={cfg.batch_size}"
                          )
        datapreprocessor = SlakhCollator(mode='full',
                                         name_to_ix=MIDIClassName2class_idx,
                                         ix_to_name=class_idx2MIDIClass,
                                         plugin_labels_num=MIDI_Class_NUM-1)
        # loss function       
    else:
        Model = getattr(TranscriptionModel, cfg.transcription.model.type)
        model = Model(cfg, **cfg.transcription.model.args)
        experiment_name = (
                          f"{cfg.transcription.model.type}-"
                          f"{cfg.transcription.backend.acoustic.type}-"
                          f"{cfg.transcription.backend.language.type}_"
                          f"{cfg.transcription.backend.language.args.hidden_size}-"        
                          f"{cfg.MIDI_MAPPING.type}-"
                          f"{cfg.inst_sampler.mode}_{cfg.inst_sampler.temp}_"
                          f"{cfg.inst_sampler.samples}p_{cfg.inst_sampler.neg_samples}"
                          f"noise{cfg.inst_sampler.audio_noise}-"
                          f"fps={cfg.transcription.model.args.frames_per_second}-"
                          f"csize={MIDI_Class_NUM}-"
                          f"bz={cfg.batch_size}"
                          )
        DataPreprocessor = End2EndBatchDataPreprocessor
        # loss function
        loss_function = get_loss_function(cfg.transcription.model.loss_types)

    # callbacks
    # save checkpoint callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')    
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint,
                                          auto_insert_metric_name=False)
    callbacks = [checkpoint_callback, lr_monitor]
    
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)

    # learning rate reduce function.
    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)

    if cfg.transcription.model.type=='Semantic_Segmentation':
        # PL model
        pl_model = BaselineTranscription(
            network=model,
            lr_lambda=lr_lambda,
            batch_data_preprocessor=datapreprocessor,
            cfg=cfg
        )        
    else:        
        # PL model
        if cfg.transcription.evaluation.checkpoint_path:
            user_input = input(f'Are you sure you want to use checkpoint at {cfg.transcription.evaluation.checkpoint_path}? [y/n]')
            
            if user_input.lower()=='y':
                print(f"continue training from previous ckpt {cfg.transcription.evaluation.checkpoint_path}")
                pl_model = Transcription.load_from_checkpoint(
                    cfg.transcription.evaluation.checkpoint_path,
                    network=model,
                    loss_function=loss_function,
                    lr_lambda=lr_lambda,
                    batch_data_preprocessor=DataPreprocessor(**cfg.transcription.batchprocess,
                                                             transcription=True,
                                                             source_separation=False),
                    cfg=cfg
                )
            else:
                raise ValueError('Please change transcription.evaluation.checkpoint_path to null if you want to train from scratch')
        else:
            pl_model = Transcription(
                network=model,
                loss_function=loss_function,
                lr_lambda=lr_lambda,
                batch_data_preprocessor=DataPreprocessor(**cfg.transcription.batchprocess,
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
    trainer.fit(pl_model, data_module)
    trainer.test(pl_model, data_module.test_dataloader())


if __name__ == '__main__':
    main()
