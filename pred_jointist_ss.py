from functools import partial
import os
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )

# from End2End.Data import DataModuleEnd2End, End2EndBatchDataPreprocessor, FullPreprocessor, WildDataset
import End2End.Data as Data
from End2End.tasks.jointist_ss import Jointist_SS

# packages for transcription
from End2End.tasks.transcription import Transcription
import End2End.models.transcription.combined as TranscriptionModel

# packages for source separation
import End2End.models.separation as SeparationModel
from End2End.tasks.separation import Separation
from End2End.tasks.t_separation import TSeparation

# packages for detection
import End2End.tasks.detection as Detection
import End2End.models.instrument_detection as DectectionModel
import End2End.models.instrument_detection.combined as CombinedModel
import End2End.models.instrument_detection.backbone as BackBone
import End2End.models.transformer as Transformer

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path




@hydra.main(config_path="End2End/config/", config_name="jointist_ss_inference")
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
    assert cfg.audio_path!=None, "Please provide your audio path before continuing."
    
    cfg.audio_path = to_absolute_path(cfg.audio_path)
    
    cfg.datamodule.waveform_hdf5s_dir = to_absolute_path(os.path.join('hdf5s', 'waveforms'))   

    cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
    cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
    cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
    cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_class/')         
    
    experiment_name = "jointist_inference"
    
    # Prepare dataset
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

    print(f"{len(pred_loader)=}")
    # get checkpoint_paths
    tseparation_ckpt = to_absolute_path(cfg.checkpoint.tseparation)            
    
    # defining transcription model
    Model = getattr(TranscriptionModel, cfg.transcription.model.type)
    model = Model(cfg, **cfg.transcription.model.args)
    transcription_model = Transcription(
        network=model,
        loss_function=None,
        lr_lambda=None,
        batch_data_preprocessor=None,
        cfg=cfg)
    
    
    model = getattr(SeparationModel, cfg.separation.model.type)\
                              (**cfg.separation.model.args, spec_cfg=cfg.separation.feature)
    
    separation_model = Separation(
        network=model,
        loss_function=None,
        lr_lambda=None,
        batch_data_preprocessor=None,
        cfg=cfg
    )            
    
    
    # defining jointist
    tseparation_model = TSeparation.load_from_checkpoint(
        tseparation_ckpt,
        transcription_model = transcription_model,
        separation_model = separation_model,
        batch_data_preprocessor = None,        
        lr_lambda=None,
        cfg=cfg
    )        

    # defining instrument detection model
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
                                         spec_args=cfg.detection.feature
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

    detection_model = getattr(Detection, cfg.detection.task).load_from_checkpoint(to_absolute_path(cfg.checkpoint.detection),
                                              network=model,
                                              lr_lambda=None,
                                              cfg=cfg,
                                              strict=True)

    experiment_name = (
                      f"Eval-Jointist-"
                      f"{cfg.datamodule.type}-"
                      )    
    
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name=experiment_name)    
    
    # defining jointist
    jointist = Jointist_SS(
        detection_model=detection_model,
        tseparation_model=tseparation_model,
        lr_lambda=None,
        cfg=cfg        
    )    
    

    # defining Trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger)

    # Fit, evaluate, and save checkpoints.
    predictions = trainer.predict(jointist, pred_loader)
#     print(predictions)
    

if __name__ == '__main__':
    main()
