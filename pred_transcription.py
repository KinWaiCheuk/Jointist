from functools import partial
import os


from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import End2End.Data as Data
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




@hydra.main(config_path="End2End/config/", config_name="pred_transcription_config")
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



    # data module
    if cfg.datamodule.type=='MSD':
        from samidata_pt.loaders.msd.loader import get_dataloader
        pred_loader = get_dataloader(**cfg.datamodule.dataloader_cfg)
        
    elif cfg.datamodule.type=='H5Dataset':
        h5_root = to_absolute_path(cfg.h5_root)
        cfg.datamodule.args.h5_path = os.path.join(h5_root, cfg.h5_name)
        dataset = getattr(Data, cfg.datamodule.type)\
            (**cfg.datamodule.args, MIDI_MAPPING=cfg.MIDI_MAPPING)
        pred_loader = DataLoader(dataset, **cfg.datamodule.dataloader_cfg.pred)
    else:
        dataset = getattr(Data, cfg.datamodule.type)\
            (**cfg.datamodule.args, MIDI_MAPPING=cfg.MIDI_MAPPING)
        pred_loader = DataLoader(dataset, **cfg.datamodule.dataloader_cfg.pred)

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
    elif cfg.datamodule.type=='H5Dataset':
        Model = getattr(TranscriptionModel, cfg.transcription.model.type)
        model = Model(cfg, **cfg.transcription.model.args)
        h5_filename = os.path.basename(cfg.datamodule.args.h5_path)
        experiment_name = (
                          f"Eval-"
                          f"{cfg.h5_name}-"
                          f"{cfg.datamodule.type}-"
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
    else:
        Model = getattr(TranscriptionModel, cfg.transcription.model.type)
        model = Model(cfg, **cfg.transcription.model.args)
        experiment_name = (
                          f"Eval-"
                          f"{cfg.datamodule.type}-"
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

    if cfg.transcription.model.type=='Semantic_Segmentation':
        # PL model       
        pl_model = BaselineTranscription.load_from_checkpoint(checkpoint_path,
                                                      network=model,
                                                      lr_lambda=None,
                                                      batch_data_preprocessor=None,
                                                      cfg=cfg)        
    else:        
        # PL model
        pl_model = Transcription.load_from_checkpoint(checkpoint_path,
                                                    network=model,
                                                    loss_function=None,
                                                    lr_lambda=None,
                                                    batch_data_preprocessor=None,
                                                    cfg=cfg)        

    if cfg.trainer.gpus==0: # If CPU is used, disable syncbatch norm
        cfg.trainer.sync_batchnorm=False    
    
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=None,
        logger=logger
    )
    
    

    # Fit, evaluate, and save checkpoints.
    trainer.predict(pl_model, pred_loader)


if __name__ == '__main__':
    main()

