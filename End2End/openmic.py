import omegaconf
import os
import pathlib
import tqdm
import h5py
from End2End.utils import int16_to_float32
import pandas as pd
import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule
from typing import Optional
import torch

class Openmic2018DataModule(LightningDataModule):
    def __init__(
        self,
        waveform_hdf5s_dir: str,
        notes_pkls_dir: str,
        dataset_cfg: omegaconf.dictconfig.DictConfig,
        dataloader_cfg: omegaconf.dictconfig.DictConfig,
        MIDI_MAPPING: omegaconf.dictconfig.DictConfig
    ):
        r"""Instrument classification data module.

        Args:
            waveform_hdf5s_dir: str
            notes_pkl_pth: str
            segment_seconds: float, e.g., 2.0
            frames_per_second: int, e.g., 100
            augmentor: Augmentor
            classes_num: int, plugins number, e.g., 167
            batch_size: int
            steps_per_epoch: int
            num_workers: int
            mini_data: bool, set True to use a small amount of data for debugging
        """
        super().__init__()    
        self.MIDI_MAPPING = MIDI_MAPPING
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.notes_pkls_dir = notes_pkls_dir
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        
        self.train_kwargs = {
            'split': 'train',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'label_csv_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.train
        }
        
        self.val_kwargs = {
            'split': 'test',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'label_csv_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.val
        }        
        
        self.test_kwargs = {
            'split': 'test',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'label_csv_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.test
        }
        

        r"""called on every device."""
        
    def setup(self, stage: Optional[str] = None):
        if stage=='fit' or stage==None:
            self.train_dataset = Openmic2018(**self.train_kwargs)
            self.val_dataset = Openmic2018(**self.val_kwargs)

        if stage=='test' or stage==None:
            self.test_dataset = Openmic2018(**self.test_kwargs)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.train
        )
        return loader
    

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.val
        )
        return loader    
    
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.test
        )
        return loader
    
class Openmic2018DataModule_npz(LightningDataModule):
    def __init__(
        self,
        waveform_hdf5s_dir: str,
        notes_pkls_dir: str,
        dataset_cfg: omegaconf.dictconfig.DictConfig,
        dataloader_cfg: omegaconf.dictconfig.DictConfig,
        MIDI_MAPPING: omegaconf.dictconfig.DictConfig
    ):
        r"""Instrument classification data module.

        Args:
            waveform_hdf5s_dir: str
            notes_pkl_pth: str
            segment_seconds: float, e.g., 2.0
            frames_per_second: int, e.g., 100
            augmentor: Augmentor
            classes_num: int, plugins number, e.g., 167
            batch_size: int
            steps_per_epoch: int
            num_workers: int
            mini_data: bool, set True to use a small amount of data for debugging
        """
        super().__init__()    
        self.MIDI_MAPPING = MIDI_MAPPING
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.notes_pkls_dir = notes_pkls_dir
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        
        self.train_kwargs = {
            'split': 'train',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'npz_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.train
        }
        
        self.val_kwargs = {
            'split': 'test',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'npz_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.val
        }        
        
        self.test_kwargs = {
            'split': 'test',
            'waveform_hdf5s_dir': waveform_hdf5s_dir,
            'npz_dir': notes_pkls_dir,
            'pre_load_audio': bool,
            'MIDI_MAPPING': MIDI_MAPPING,
            **dataset_cfg.test
        }
        

        r"""called on every device."""
        
    def setup(self, stage: Optional[str] = None):
        if stage=='fit' or stage==None:
            self.train_dataset = Openmic2018_npz(**self.train_kwargs)
            self.val_dataset = Openmic2018_npz(**self.val_kwargs)

        if stage=='test' or stage==None:
            self.test_dataset = Openmic2018_npz(**self.test_kwargs)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.train
        )
        return loader
    

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.val
        )
        return loader    
    
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            collate_fn=collate_fn,
            **self.dataloader_cfg.test
        )
        return loader    

class Openmic2018:
    def __init__(
        self,
        split: str,
        waveform_hdf5s_dir: str,
        label_csv_dir: str,
        pre_load_audio: bool,
        slack_mapping: bool,
        MIDI_MAPPING: omegaconf.dictconfig.DictConfig,
    ):
        r"""Loading openemic2018 dataset
        references 
        https://github.com/keunwoochoi/openmic-2018-tfrecord/blob/master/openmic2018.py
        https://github.com/cosmir/openmic-2018
        https://zenodo.org/record/1432913

        Args:
            split: either `train` or `test`
            waveform_hdf5s_dir: path to the hdf5s files
            label_csv_dir: path to the label csv file, keep using the old argument name to be compatitlbe with slakh
            pre_load_audio: If `True`, load audio into the RAM. Otherwise, load audio on-the-fly
            MIDI_MAPPING: Omega dictionaries for mapping instrument names to indice and vice versa
        """
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.label_csv_dir = label_csv_dir
        self.slack_mapping = slack_mapping
            
        self.name_to_ix = MIDI_MAPPING.NAME_TO_IX
        self.ix_to_name = MIDI_MAPPING.IX_TO_NAME
        self.plugin_labels_num = MIDI_MAPPING.plugin_labels_num # include the empty class to make it consistent with Slakh      
        
        self.pre_load_audio = pre_load_audio
        
        split_hdf5s_dir = os.path.join(waveform_hdf5s_dir, split)
        hdf5_paths = sorted([str(path) for path in pathlib.Path(split_hdf5s_dir).rglob('*.h5')])

        self.audio_name_list = []
        # Load audio files into the RAM to speed things up
        if pre_load_audio==True:
            for hdf5_path in tqdm.tqdm(hdf5_paths, desc=f'Loading {split} hdf5 files'):
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        if hf.attrs['split'].decode() == split:
                            audio_name = str(os.path.splitext(hf.attrs['audio_name'])[0].decode())
                            waveform = int16_to_float32(hf['waveform'][()])
                            self.audio_name_list.append([hf.attrs['split'].decode(), audio_name, waveform])
                except Exception as e:
                    print(e)
        elif pre_load_audio==False:
            for hdf5_path in tqdm.tqdm(hdf5_paths, desc=f'Loading {split} hdf5 files'):
                self.audio_name_list.append([split, hdf5_path])
        else:
            raise ValueError(f'pre_load_audio={pre_load_audio} is not supported')
            
        self.label_csv = pd.read_csv(os.path.join(self.label_csv_dir, 'openmic-2018-aggregated-labels.csv'), header=0)
            
        
    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        r"""Get input and target for training.

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'hdf5_name': str,
            'plugin_names': list,
            'instruments': (classes_num),
        """
        if self.pre_load_audio:
            split, hdf5_name, waveform = self.audio_name_list[idx]
        else:
            split, hdf5_path = self.audio_name_list[idx]
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    hdf5_name = str(os.path.splitext(hf.attrs['audio_name'])[0].decode())
                    waveform = int16_to_float32(hf['waveform'][()])            

        data_dict = {}       

        # Load segment waveform.
        # with h5py.File(waveform_hdf5_path, 'r') as hf:
        audio_length = len(waveform)

        data_dict['waveform'] = waveform
        data_dict['hdf5_name'] = hdf5_name        

        mask = self.label_csv[self.label_csv.sample_key == hdf5_name].relevance>0.5 # masking out negative labels
        unique_plugin_names = self.label_csv[self.label_csv.sample_key == hdf5_name][mask].instrument.to_numpy()         
        data_dict['plugin_names'] = unique_plugin_names
    
        target = np.zeros(self.plugin_labels_num)  # (plugin_names_num,)
        # capitalize the first letter using .title()
        # For example 'accordion'.title() = 'Accordion'
        plugin_ids = []
        for plugin_name in unique_plugin_names:
            if self.slack_mapping:
                if plugin_name=='synthesizer':
                    plugin_name='Synth Lead'
                elif plugin_name in ['banjo', 'mandolin', 'ukulele', 'guitar']:
                    plugin_name='Acoustic Guitar'                
                elif plugin_name in ['drums', 'cymbals']:
                    plugin_name='Empty'
                elif plugin_name in ['mallet_percussion']:
                    plugin_name='Chromatic Percussion'
                plugin_ids.append(self.name_to_ix[plugin_name.title()])
            else:
                plugin_ids.append(self.name_to_ix[plugin_name])
        for plugin_id in plugin_ids:          
            target[plugin_id] = 1
        data_dict['instruments'] = target    

        return data_dict
    
    
class Openmic2018_npz:
    def __init__(
        self,
        split: str,
        waveform_hdf5s_dir: str,
        npz_dir: str,
        pre_load_audio: bool,
        slack_mapping: bool,
        MIDI_MAPPING: omegaconf.dictconfig.DictConfig,
    ):
        r"""Loading openemic2018 dataset
        references 
        https://github.com/keunwoochoi/openmic-2018-tfrecord/blob/master/openmic2018.py
        https://github.com/cosmir/openmic-2018
        https://zenodo.org/record/1432913

        Args:
            split: either `train` or `test`
            waveform_hdf5s_dir: path to the hdf5s files
            label_csv_dir: path to the label csv file, keep using the old argument name to be compatitlbe with slakh
            pre_load_audio: If `True`, load audio into the RAM. Otherwise, load audio on-the-fly
            MIDI_MAPPING: Omega dictionaries for mapping instrument names to indice and vice versa
        """
               
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.npz_dir = npz_dir
        data = np.load(os.path.join(self.npz_dir, 'openmic-2018.npz'), allow_pickle=True)
        self.Y_true = data['Y_true']
        self.Y_true = self.Y_true >= 0.5
        self.Y_mask = data['Y_mask']
        self.sample_key = data['sample_key']
        
        self.slack_mapping = slack_mapping
            
        self.name_to_ix = MIDI_MAPPING.NAME_TO_IX
        self.ix_to_name = MIDI_MAPPING.IX_TO_NAME
        self.plugin_labels_num = MIDI_MAPPING.plugin_labels_num # include the empty class to make it consistent with Slakh      
        
        self.pre_load_audio = pre_load_audio
        
        split_hdf5s_dir = os.path.join(waveform_hdf5s_dir, split)
        hdf5_paths = sorted([str(path) for path in pathlib.Path(split_hdf5s_dir).rglob('*.h5')])

        self.audio_name_list = []
        # Load audio files into the RAM to speed things up
        if pre_load_audio==True:
            for hdf5_path in tqdm.tqdm(hdf5_paths, desc=f'Loading {split} hdf5 files'):
                try:
                    with h5py.File(hdf5_path, 'r') as hf:
                        if hf.attrs['split'].decode() == split:
                            audio_name = str(os.path.splitext(hf.attrs['audio_name'])[0].decode())
                            waveform = int16_to_float32(hf['waveform'][()])
                            self.audio_name_list.append([hf.attrs['split'].decode(), audio_name, waveform])
                except Exception as e:
                    print(e)
        elif pre_load_audio==False:
            for hdf5_path in tqdm.tqdm(hdf5_paths, desc=f'Loading {split} hdf5 files'):
                self.audio_name_list.append([split, hdf5_path])
        else:
            raise ValueError(f'pre_load_audio={pre_load_audio} is not supported')
            

            
        
    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        r"""Get input and target for training.

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'hdf5_name': str,
            'plugin_names': list,
            'instruments': (classes_num),
        """
        if self.pre_load_audio:
            split, hdf5_name, waveform = self.audio_name_list[idx]
        else:
            split, hdf5_path = self.audio_name_list[idx]
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    hdf5_name = str(os.path.splitext(hf.attrs['audio_name'])[0].decode())
                    waveform = int16_to_float32(hf['waveform'][()])            

        data_dict = {}       

        # Load segment waveform.
        # with h5py.File(waveform_hdf5_path, 'r') as hf:
        audio_length = len(waveform)

        data_dict['waveform'] = waveform
        data_dict['hdf5_name'] = hdf5_name        
        npz_idx = np.where(self.sample_key==hdf5_name)[0]
        data_dict['instruments'] = self.Y_true[npz_idx[0]]
        data_dict['mask'] = self.Y_mask[npz_idx[0]]

        return data_dict    
    
    
def collate_fn(list_data_dict):
    r"""Collate input and target of segments to a mini-batch.

    Args:
        list_data_dict: e.g. [
            {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...},
            {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...},
            ...]

    Returns:
        data_dict: e.g. {
            'waveform': (batch_size, segment_samples)
            'frame_roll': (batch_size, segment_frames, classes_num),
            ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        if key in ['plugin_names']:
            np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        elif key=='hdf5_name':
            np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        else:
            np_data_dict[key] = torch.Tensor(np.array([data_dict[key] for data_dict in list_data_dict]))

    return np_data_dict    