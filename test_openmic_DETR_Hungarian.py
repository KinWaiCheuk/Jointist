from functools import partial
import os

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from End2End.openmic import Openmic2018DataModule
from End2End.Task import DETR_IR
from End2End.MIDI_program_map import (MIDI_PROGRAM_NUM,
                                      MIDIProgramName2class_idx,
                                      class_idx2MIDIProgramName,
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      W_MIDI_Class_NUM,
                                      W_MIDIClassName2class_idx,
                                      W_class_idx2MIDIClass,
                                      )
import End2End.models.detr as DETR_Model

from jointist.config import (
    BEGIN_NOTE,
    PLUGIN_LABELS_NUM,
    FRAMES_PER_SECOND,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    VELOCITY_SCALE,
    TAGGING_SEGMENT_SECONDS,
    PLUGIN_NAME_TO_INSTRUMENT,
    PLUGIN_LB_TO_IX,
    PLUGIN_IX_TO_LB
)

from jointist.data.augmentors import Augmentor
from jointist.lr_schedulers import get_lr_lambda
# from jointist.models.instruments_classification_models import get_model_class

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

@hydra.main(config_path="End2End/config/", config_name="openmic-DETR_Hungarian_IR")
def main(cfg):
    
    cfg.datamodule.waveform_hdf5s_dir = to_absolute_path(os.path.join('hdf5s', 'openmic_waveforms'))
    
    if cfg.MIDI_MAPPING.type=='plugin_names':
        cfg.MIDI_MAPPING.plugin_labels_num = PLUGIN_LABELS_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = PLUGIN_LB_TO_IX
        cfg.MIDI_MAPPING.IX_TO_NAME = PLUGIN_IX_TO_LB
        cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes3/')   
    elif cfg.MIDI_MAPPING.type=='MIDI_programs':
        cfg.MIDI_MAPPING.plugin_labels_num = MIDI_PROGRAM_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = MIDIProgramName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIProgramName
        cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_instrument/')
    elif cfg.MIDI_MAPPING.type=='MIDI_class':
        cfg.MIDI_MAPPING.plugin_labels_num = MIDI_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = MIDIClassName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = class_idx2MIDIClass
        cfg.datamodule.notes_pkls_dir = to_absolute_path('datasets/openmic-2018')         
    elif cfg.MIDI_MAPPING.type=='W_MIDI_class':
        cfg.MIDI_MAPPING.plugin_labels_num = W_MIDI_Class_NUM
        cfg.MIDI_MAPPING.NAME_TO_IX = W_MIDIClassName2class_idx
        cfg.MIDI_MAPPING.IX_TO_NAME = W_class_idx2MIDIClass
        cfg.datamodule.notes_pkls_dir = to_absolute_path('instruments_classification_notes_MIDI_class/')              
        

    experiment_name = (
                      f"Decoder_L{cfg.model.args.num_decoder_layers}-"
                      f"empty_{cfg.model.eos_coef}-"
                      f"feature_weigh_{cfg.model.args.feature_weight}-"
                      f"{cfg.model.type}-"
                      f"hidden={cfg.model.args.hidden_dim}-"
                      f"Q={cfg.model.args.num_Q}-"
                      f"LearnPos={cfg.model.args.learnable_pos}-"
                      f"aux_loss-bsz={cfg.batch_size}-"
                      f"audio_len={cfg.segment_seconds}"
                      )



    # data module    # augmentor
    augmentor = Augmentor(augmentation=cfg.augmentation) if cfg.augmentation else None
    data_module = Openmic2018DataModule(**cfg.datamodule, MIDI_MAPPING=cfg.MIDI_MAPPING)
#     data_module.setup()        

    # model
#     Model = getattr(IR_Model, cfg.model.type)
    Model = getattr(DETR_Model, cfg.model.type)
    model = Model(num_classes=cfg.MIDI_MAPPING.plugin_labels_num, **cfg.model.args)
#     model = Model(classes_num=cfg.MIDI_MAPPING.plugin_labels_num)

    # PL model
    pl_model = DETR_IR.load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                            network=model,
                                            learning_rate=cfg.lr,
                                            lr_lambda=None,
                                            cfg=cfg)

    trainer = pl.Trainer(
        **cfg.trainer,
    )
    

    # Fit, evaluate, and save checkpoints.
    trainer.test(pl_model, data_module)

if __name__ == "__main__":
    main()