from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.nn as nn

from End2End.Data import DataModuleEnd2End
import End2End.tasks.detection as Detection
from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
# import End2End.models.instrument_detection as DectectionModel
from End2End.models.instrument_detection.CLS import CNNSA
import End2End.models.instrument_detection.combined as CombinedModel
from End2End.models.instrument_detection.openmic_baseline import DecisionLevelSingleAttention
import End2End.models.instrument_detection.backbone as BackBone
import End2End.models.transformer as Transformer

from End2End.data.augmentors import Augmentor
from End2End.lr_schedulers import get_lr_lambda
# from jointist.models.instruments_classification_models import get_model_class

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path

@hydra.main(config_path="End2End/config/", config_name="detection_config")
def main(cfg):
    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None
    # data module    
    
    if cfg.MIDI_MAPPING.type=='openmic':
        print(f"------training on openmic---------------")
        from End2End.Openmic_map import (
                                          OpenMic_Class_NUM,
                                          Name2OpenmicIDX,
                                          OpenmicIDX2Name,
                                          )
        from End2End.openmic import Openmic2018DataModule, Openmic2018DataModule_npz
        cfg.datamodule.waveform_hdf5s_dir = to_absolute_path(os.path.join('hdf5s', 'openmic_waveforms'))
        cfg.datamodule.notes_pkls_dir = to_absolute_path(os.path.join('datasets', 'openmic-2018'))
        cfg.MIDI_MAPPING.plugin_labels_num = OpenMic_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = Name2OpenmicIDX
        cfg.MIDI_MAPPING.IX_TO_NAME = OpenmicIDX2Name
        data_module = Openmic2018DataModule_npz(**cfg.datamodule, MIDI_MAPPING=cfg.MIDI_MAPPING)          
        
    elif cfg.MIDI_MAPPING.type=='slakh':
        cfg.datamodule.waveform_dir = to_absolute_path(cfg.datamodule.waveform_dir)
        cfg.datamodule.pkl_dir = to_absolute_path(cfg.datamodule.pkl_dir)
        cfg.datamodule.slakhdata_root = to_absolute_path(cfg.datamodule.slakhdata_root)
        cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
        data_module = DataModuleEnd2End(**cfg.datamodule,augmentor=augmentor, MIDI_MAPPING=cfg.MIDI_MAPPING)
    #     data_module.setup()         
    else:
        raise ValueError(f"Please choose the correct MIDI_MAPPING type. {cfg.MIDI_MAPPING.type=} is not defined")
      
    if cfg.detection.type=='CombinedModel_Linear':
        experiment_name = (
                          f"{cfg.detection.type}-{cfg.detection.task}-{cfg.detection.backbone.type}"
                          f"hidden={cfg.detection.transformer.hidden_dim}-"
                          f"aux_loss-bsz={cfg.batch_size}-"
                          f"audio_len={cfg.segment_seconds}"
                          )
    elif 'CombinedModel_CLS' in cfg.detection.type:
        experiment_name = (
                          f"{cfg.detection.type}-{cfg.detection.task}-{cfg.detection.backbone.type}-"
                          f"bsz={cfg.batch_size}-"
                          f"audio_len={cfg.segment_seconds}"
                          )
    elif 'CombinedModel_NewCLS' in cfg.detection.type:
        experiment_name = (
                          f"{cfg.detection.type}-{cfg.detection.task}-{cfg.detection.backbone.type}-"
                          f"En_L{cfg.detection.transformer.args.num_encoder_layers}-"            
                          f"bsz={cfg.batch_size}-"
                          f"audio_len={cfg.segment_seconds}"
                          )     
    elif 'Original' in cfg.detection.type:
        experiment_name = (
                          f"{cfg.detection.type}-{cfg.detection.backbone.type}-"
                          f"bsz={cfg.batch_size}-"
                          f"audio_len={cfg.segment_seconds}"
                          )
    elif 'CombinedModel_A' in cfg.detection.type:
        experiment_name = (
                          f"pos_Decoder-ignore_padding"
                          f"{cfg.detection.type}-{cfg.detection.task}-"
                          f"{cfg.detection.backbone.type}_{cfg.detection.transformer.type}-"
                          f"En_L{cfg.detection.transformer.args.num_encoder_layers}-"
                          f"De_L{cfg.detection.transformer.args.num_decoder_layers}-"
    #                       f"empty_{cfg.model.eos_coef}-"
    #                       f"feature_weigh_{cfg.model.args.feature_weight}-"
                          f"hidden={cfg.detection.transformer.args.d_model}-"            
                          f"TDrop={cfg.detection.transformer.args.dropout}-"
                          f"TarShu={cfg.detection.model.shuffle_target}-"
                          f"s_logit={cfg.detection.model.scale_logits}-"
                          f"TarDrop={cfg.detection.model.target_dropout}-"
                          f"aux_loss-bsz={cfg.batch_size}-"
                          )
    elif cfg.detection.type=='OpenMicBaseline':
        experiment_name = (
                          f"OpenMicBaseline"
                          )        
    else:
        experiment_name = (
                          f"{cfg.detection.type}-{cfg.detection.task}-"
                          f"{cfg.detection.backbone.type}_{cfg.detection.transformer.type}-"
                          f"En_L{cfg.detection.transformer.args.num_encoder_layers}-"
                          f"De_L{cfg.detection.transformer.args.num_decoder_layers}-"
    #                       f"empty_{cfg.model.eos_coef}-"
    #                       f"feature_weigh_{cfg.model.args.feature_weight}-"
                          f"hidden={cfg.detection.transformer.args.d_model}-"
                          f"TDrop={cfg.detection.transformer.args.dropout}-"
                          f"aux_loss-bsz={cfg.batch_size}-"
                          )
   

    # model
    if cfg.detection.type!='OpenMicBaseline': # only need backbone when doing transformer based models
        backbone = getattr(BackBone, cfg.detection.backbone.type)(**cfg.detection.backbone.args)
    
    
    if cfg.detection.type=='CombinedModel_Linear':
        linear = nn.Linear(cfg.detection.transformer.hidden_dim*15*3, cfg.detection.transformer.hidden_dim)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         linear=linear,
                                         spec_args=cfg.detection.feature
                                         )
    elif 'CombinedModel_CLS' in cfg.detection.type:
        encoder = getattr(Transformer, cfg.detection.transformer.type)(cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         encoder=encoder,
                                         spec_args=cfg.detection.feature
                                         )
    elif 'CombinedModel_NewCLS' in cfg.detection.type:
        encoder = getattr(Transformer, cfg.detection.transformer.type)(**cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         encoder=encoder,
                                         spec_args=cfg.detection.feature
                                         )        
    elif 'Original' in cfg.detection.type:
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         spec_args=cfg.detection.feature
                                         )
    elif 'CombinedModel_A' in cfg.detection.type:
        transformer = nn.Transformer(**cfg.detection.transformer.args)
        model = getattr(CombinedModel, cfg.detection.type)(
                                         cfg.detection.model,
                                         backbone=backbone,
                                         transformer=transformer,
                                         spec_args=cfg.detection.feature
                                         )
    elif cfg.detection.type=='OpenMicBaseline':
        model = DecisionLevelSingleAttention(
                                         **cfg.detection.model.args,
                                         spec_args=cfg.detection.feature
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
                                         spec_args=cfg.detection.feature
                                         )

    # callbacks
    # save checkpoint callback    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint, auto_insert_metric_name=False)    
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Defining a tensorboard logger
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)

    # learning rate reduce function.
    lr_lambda = partial(get_lr_lambda, **cfg.scheduler.args)

    # PL model
    pl_model = getattr(Detection, cfg.detection.task)(
        network=model,
        lr_lambda=lr_lambda,
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
    trainer.test(pl_model, data_module)

if __name__ == "__main__":
    main()