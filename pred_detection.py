from functools import partial
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor, FullPreprocessor, WildDataset
import End2End.tasks.detection as Detection
from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
# import End2End.models.instrument_detection as DectectionModel
from End2End.models.instrument_detection.CLS import CNNSA
import End2End.models.instrument_detection.combined as CombinedModel
import End2End.models.instrument_detection.backbone as BackBone
import End2End.models.transformer as Transformer

from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
# from jointist.models.instruments_classification_models import get_model_class

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

def show_instrument_name(plugin_idxs):
    instrument_name = []
    for i in plugin_idxs.cpu():
        instrument_name.append(class_idx2MIDIClass[i.item()])
    return instrument_name


@hydra.main(config_path="End2End/config/", config_name="detection_config")
def main(cfg):
            
    cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
    cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
    cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass

    experiment_name = ("ITW_audio_evaluation")

    # data module    # augmentor
    dataset = WildDataset(**cfg.datamodule.wild, MIDI_MAPPING=cfg.MIDI_MAPPING)
    pred_loader = DataLoader(dataset, **cfg.datamodule.dataloader_cfg.pred) 
#     data_module.setup()        

    # model
    if cfg.detection.type!='OpenMicBaseline': # only need backbone when doing transformer based models
        backbone = getattr(BackBone, cfg.detection.backbone.type)(**cfg.detection.backbone.args)
    
    
    if cfg.detection.type=='CombinedModel_Linear':
        linear = nn.Linear(cfg.detection.transformer.hidden_dim*15*3, cfg.detection.transformer.hidden_dim)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         linear=linear,
                                         spec_args=cfg.feature
                                         )
    elif 'CombinedModel_CLS' in cfg.detection.type:
        encoder = getattr(Transformer, cfg.detection.transformer.type)(cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         encoder=encoder,
                                         spec_args=cfg.feature
                                         )
    elif 'CombinedModel_NewCLS' in cfg.detection.type:
        encoder = getattr(Transformer, cfg.detection.transformer.type)(**cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         encoder=encoder,
                                         spec_args=cfg.feature
                                         )        
    elif 'Original' in cfg.detection.type:
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         spec_args=cfg.feature
                                         )
    elif 'CombinedModel_A' in cfg.detection.type:
        transformer = nn.Transformer(**cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         transformer=transformer,
                                         spec_args=cfg.feature
                                         )
    elif cfg.detection.type=='OpenMicBaseline':
        model = DecisionLevelSingleAttention(
                                         **cfg.detection.model.args,
                                         spec_args=cfg.feature
                                         )        
    else:
        if cfg.detection.transformer.type=='torch_Transformer_API':
            print(f"using torch transformer")
            transformer = nn.Transformer(**cfg.detection.transformer.args)
        else:
            transformer = getattr(Transformer, cfg.detection.transformer.type)(**cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         transformer=transformer,
                                         spec_args=cfg.feature
                                         )

    # learning rate reduce function.
    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)    
    
    # Defining a tensorboard logger
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)    
    
    # PL model
    # PL model
    pl_model = getattr(Detection, cfg.detection.task).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                              network=model,
                                              lr_lambda=lr_lambda,
                                              cfg=cfg,
                                              strict=False)

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger
    )
    

    # Fit, evaluate, and save checkpoints.
    predictions = trainer.predict(pl_model, pred_loader)
    
#     for i in predictions:
#         print(show_instrument_name(i))


if __name__ == "__main__":
    main()