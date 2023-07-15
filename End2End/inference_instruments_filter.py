import argparse

import torch
import pretty_midi
import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pathlib
import pickle
import sklearn
import scipy.stats
import soundfile
import os
import time
import matplotlib.pyplot as plt
import yaml

from End2End.constants import (
    SAMPLE_RATE,
    PLUGIN_LABELS_NUM,
    BEGIN_NOTE,
    PLUGIN_LABELS,
    CLASSES_NUM,
    TAGGING_SEGMENT_SECONDS,
    FRAMES_PER_SECOND,
    VELOCITY_SCALE,
    PLUGIN_LB_TO_IX,
    PLUGIN_IX_TO_LB,
    PLUGIN_NAME_TO_INSTRUMENT,
)
from End2End.piano_vad import (
    note_detection_with_onset_offset_regress,
    note_detection_with_onset_regress,
    pedal_detection_with_onset_offset_regress,
    drums_detection_with_onset_regress,
)
# from jointist.models.instruments_filter_models import get_model_class
from End2End.utils import read_yaml, int16_to_float32
from End2End.data.data_modules import get_single_note_onset_roll

'''
class MusicTranscriber:
    def __init__(
        self,
        programs_list,
        model_type: str,
        modeling_offset: bool,
        modeling_velocity: bool = False,
        checkpoint_path: str = '',
        segment_samples: int = SAMPLE_RATE * 10,
        batch_size: int = 12,
        device: str = 'cuda',
    ):
        r"""MusicTranscriber is used to transcribe an audio clip into MIDI
        events and write out to a MIDI file.

        Args:
            programs (list of str): e.g., ['0', '16', '33', '48', 'percussion']
            model_type (str)
            modeling_velocity (bool): whether to model velocity or not
            checkpoint_path (str)
            segment_samples (int): an audio clip is split into segments
                (such as 10-second segments) to the trained system.
            batch_size (int): number of segments in a mini-batch for inference
            device: 'cuda' | 'cpu'
        """
        self.programs_list = programs_list
        self.modeling_offset = modeling_offset
        self.segment_samples = segment_samples
        self.batch_size = batch_size
        self.programs = programs_list

        if 'cuda' in str(device) and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print('Using {} for inference.'.format(self.device))

        self.sample_rate = SAMPLE_RATE
        self.frames_per_second = FRAMES_PER_SECOND
        self.classes_num = CLASSES_NUM
        self.onset_threshold = 0.1
        self.offset_threshod = 0.1
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Initialize model.
        Model = get_model_class(model_type)

        self.model = Model(
            frames_per_second=self.frames_per_second,
            classes_num=self.classes_num,
            modeling_offset=modeling_offset,
            modeling_velocity=modeling_velocity,
        )

        # Load checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)

        # Move model to device.
        if 'cuda' in str(self.device):
            self.model.to(self.device)

    def transcribe(self, audio, plugin_id_list, midi_path):
        r"""Transcribe an audio clip into MIDI events and write out to a MIDI
        file.

        Args:
            audio: ndarray, (segment_samples,)
            midi_path: str

        Returns:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
        """
        midi_events = {}

        output_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []

        for plugin_id in plugin_id_list:
            condition = np.zeros(PLUGIN_LABELS_NUM)
            condition[plugin_id] = 1

            # --- 1. Predict probabilities ---
            print('--- 1. Predict probabilities ---')
            _output_dict = self.predict_probabilities(audio, condition)
            # os.makedirs('debug', exist_ok=True)
            # pickle.dump(output_dict, open('debug/output_dict.pkl', 'wb'))

            # Uncomment the following line to load output_dict
            # output_dict = pickle.load(open('debug/output_dict.pkl', 'rb'))

            for key in ['reg_onset_output', 'frame_output']:
                output_dict[key].append(_output_dict[key])

        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = np.concatenate(output_dict[key], axis=-1)

        # --- 2. Postprocess probabilities to MIDI events ---
        print('--- 2. Postprocess probabilities to MIDI events ---')
        midi_events = self.postprocess_probabilities_to_midi_events(output_dict)
        pickle.dump(midi_events, open('debug/midi_events.pkl', 'wb'))

        # Uncomment the following line to load midi_events
        # output_dict = pickle.load(open('debug/midi_events.pkl', 'rb'))

        # --- 3. Write MIDI events to audio ---
        print('--- 3. Write MIDI events to audio ---')
        self.write_midi_events_to_midi_file(midi_events, midi_path)

        return midi_events

    def predict_probabilities(self, audio, condition):
        r"""Predict transcription probabilities of an audio clip.

        Args:
            audio: (audio_samples,)
            midi_path: str, path to write out transcribed MIDI file.

        Returns:
            output_dict, dict: e.g., {
                'frame_output': (N, 88*5),
                'reg_onset_output': (N, 88*5),
                ...}
        """
        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples.
        audio_length = audio.shape[1]
        pad_len = int(np.ceil(audio_length / self.segment_samples)) * self.segment_samples - audio_length

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments.
        segments = self.enframe(audio, self.segment_samples)
        # (N, segment_samples)

        conditions = np.tile(condition, (len(segments), 1))

        # Inference on segments.
        output_dict = self._forward_mini_batches(self.model, segments, conditions, batch_size=self.batch_size)
        # {'frame_output': (segments_num, segment_frames, classes_num),
        #   'reg_onset_output': (segments_num, segment_frames, classes_num),
        #   ...}

        audio_duration = audio_length / self.sample_rate
        frames_num = int(audio_duration * self.frames_per_second)

        # Deframe to original length.
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:frames_num]
        # {'frame_output': (N, 88*5),
        #  'reg_onset_output': (N, 88*5),
        #  ...}

        return output_dict

    def postprocess_probabilities_to_midi_events(self, output_dict):
        r"""Postprocess probabilities to MIDI events using thresholds.

        Args:
            output_dict, dict: e.g., {
                'frame_output': (N, 88*5),
                'reg_onset_output': (N, 88*5),
                ...}

        Returns:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
        """
        post_processor = RegressionPostProcessor(
            self.frames_per_second,
            classes_num=self.classes_num,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshod,
            frame_threshold=self.frame_threshold,
            pedal_offset_threshold=self.pedal_offset_threshold,
            modeling_offset=self.modeling_offset,
        )

        programs_output_dict = {}

        for k, program in enumerate(self.programs):
            programs_output_dict[program] = {}
            for key in output_dict.keys():
                programs_output_dict[program][key] = output_dict[key][
                    :, k * self.classes_num : (k + 1) * self.classes_num
                ]
        # {'0': {
        #     'frame_output': (N, 88),
        #     'reg_onset_output': (N, 88),
        #     ...},
        #  'percussion': {
        #     'frame_output': (N, 88),
        #     'reg_onset_output': (N, 88),
        #     ...},
        #  ...}

        midi_events = {}
        for k, program in enumerate(self.programs):
            print('program: {}'.format(program))

            if program == 'percussion':
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[program],
                    detect_type='percussion',
                )

            else:
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[program],
                    detect_type='piano',
                )
            midi_events[program] = est_note_events

        return midi_events

    def write_midi_events_to_midi_file(self, midi_events, midi_path):
        r"""Write MIDI events to a MIDI file.

        Args:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
            midi_path: str, path to write out the MIDI file

        Returns:
            None
        """
        new_midi_data = pretty_midi.PrettyMIDI()

        for k, program in enumerate(self.programs):
            if program == 'percussion':
                new_track = pretty_midi.Instrument(program=0)
                new_track.is_drum = True
            else:
                new_track = pretty_midi.Instrument(program=int(program))

            new_track.name = program
            est_note_events = midi_events[program]

            for note_event in est_note_events:
                # print(note_event['offset_time'] - note_event['onset_time'])
                if note_event['offset_time'] - note_event['onset_time'] >= 0.05:
                    new_note = pretty_midi.Note(
                        pitch=note_event['midi_note'],
                        start=note_event['onset_time'],
                        end=note_event['offset_time'],
                        velocity=note_event['velocity'],
                    )
                    new_track.notes.append(new_note)
            new_midi_data.instruments.append(new_track)

        os.makedirs(os.path.dirname(midi_path), exist_ok=True)
        new_midi_data.write(midi_path)
        print('Write out transcribed result to {}'.format(midi_path))

    def enframe(self, x, segment_samples):
        r"""Enframe long sequence into short segments.

        Args:
            x: (1, audio_samples)
            segment_samples: int, e.g., 10 * sample_rate

        Returns:
            batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        r"""Deframe predicted segments to original sequence.

        Args:
            x: (N, segment_frames, classes_num)

        Returns:
            y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0:-1, :]
            # Remove an extra frame in the end of each segment caused by the
            # 'center=True' argument when calculating spectrogram.
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y

    def _forward_mini_batches(self, model, x, conditions, batch_size):
        r"""Forward data to model in mini-batch.

        Args:
            model: nn.Module
            x: ndarray, (N, segment_samples)
            batch_size: int

        Returns:
            output_dict: dict, e.g. {
                'frame_output': (segments_num, frames_num, classes_num),
                'onset_output': (segments_num, frames_num, classes_num),
                ...}
        """
        output_dict = {}

        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = x[pointer : pointer + batch_size]
            batch_condition = conditions[pointer : pointer + batch_size]

            if 'cuda' in self.device:
                batch_waveform = torch.Tensor(batch_waveform).to(self.device)
                batch_condition = torch.Tensor(batch_condition).to(self.device)

            pointer += batch_size

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_waveform, batch_condition)

            for key in batch_output_dict.keys():
                self._append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def _append_to_dict(self, dict, key, value):
        if key in dict.keys():
            dict[key].append(value)
        else:
            dict[key] = [value]
'''

class MusicTranscriber:
    def __init__(
        self,
        condition_size,
        # programs_list,
        plugin_names,
        model_type: str,
        modeling_offset: bool,
        modeling_velocity: bool = False,
        checkpoint_path: str = '',
        segment_samples: int = SAMPLE_RATE * 10,
        batch_size: int = 12,
        device: str = 'cuda',
    ):
        r"""MusicTranscriber is used to transcribe an audio clip into MIDI
        events and write out to a MIDI file.

        Args:
            programs (list of str): e.g., ['0', '16', '33', '48', 'percussion']
            model_type (str)
            modeling_velocity (bool): whether to model velocity or not
            checkpoint_path (str)
            segment_samples (int): an audio clip is split into segments
                (such as 10-second segments) to the trained system.
            batch_size (int): number of segments in a mini-batch for inference
            device: 'cuda' | 'cpu'
        """
        # self.programs_list = programs_list
        self.plugin_names = plugin_names
        self.modeling_offset = modeling_offset
        self.segment_samples = segment_samples
        self.batch_size = batch_size
        self.plugin_names = plugin_names
        # self.programs = programs_list

        if 'cuda' in str(device) and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print('Using {} for inference.'.format(self.device))

        self.sample_rate = SAMPLE_RATE
        self.frames_per_second = FRAMES_PER_SECOND
        self.classes_num = CLASSES_NUM
        self.onset_threshold = 0.1
        self.offset_threshod = 0.1
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Initialize model.
        Model = get_model_class(model_type)

        self.model = Model(
            frames_per_second=self.frames_per_second,
            condition_size=condition_size,
            classes_num=self.classes_num,
            modeling_offset=modeling_offset,
            modeling_velocity=modeling_velocity,
        )

        # Load checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)

        # Move model to device.
        if 'cuda' in str(self.device):
            self.model.to(self.device)

    def transcribe(self, audio, conditions, midi_path):
        r"""Transcribe an audio clip into MIDI events and write out to a MIDI
        file.

        Args:
            audio: ndarray, (segment_samples,)
            midi_path: str

        Returns:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
        """
        midi_events = {}

        output_dict = {}
        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = []

        for condition in conditions:

            # --- 1. Predict probabilities ---
            print('--- 1. Predict probabilities ---')
            _output_dict = self.predict_probabilities(audio, condition)
            # os.makedirs('debug', exist_ok=True)
            # pickle.dump(output_dict, open('debug/output_dict.pkl', 'wb'))

            # Uncomment the following line to load output_dict
            # output_dict = pickle.load(open('debug/output_dict.pkl', 'rb'))

            for key in ['reg_onset_output', 'frame_output']:
                output_dict[key].append(_output_dict[key])

            # from IPython import embed; embed(using=False); os._exit(0)
            # plt.matshow(output_dict['frame_output'][0][0:2000].T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
            # plt.savefig('_zz.pdf') 

        for key in ['reg_onset_output', 'frame_output']:
            output_dict[key] = np.concatenate(output_dict[key], axis=-1)

        # from IPython import embed; embed(using=False); os._exit(0) 

        # --- 2. Postprocess probabilities to MIDI events ---
        print('--- 2. Postprocess probabilities to MIDI events ---')
        midi_events = self.postprocess_probabilities_to_midi_events(output_dict)
        pickle.dump(midi_events, open('debug/midi_events.pkl', 'wb'))
        
        # Uncomment the following line to load midi_events
        # midi_events = pickle.load(open('debug/midi_events.pkl', 'rb'))

        # --- 3. Write MIDI events to audio ---
        print('--- 3. Write MIDI events to audio ---')
        self.write_midi_events_to_midi_file(midi_events, midi_path)

        return midi_events

    def predict_probabilities(self, audio, condition):
        r"""Predict transcription probabilities of an audio clip.

        Args:
            audio: (audio_samples,)
            midi_path: str, path to write out transcribed MIDI file.

        Returns:
            output_dict, dict: e.g., {
                'frame_output': (N, 88*5),
                'reg_onset_output': (N, 88*5),
                ...}
        """
        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples.
        audio_length = audio.shape[1]
        pad_len = int(np.ceil(audio_length / self.segment_samples)) * self.segment_samples - audio_length

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments.
        segments = self.enframe(audio, self.segment_samples)
        # (N, segment_samples)

        conditions = np.tile(condition, (len(segments), 1))

        # Inference on segments.
        output_dict = self._forward_mini_batches(self.model, segments, conditions, batch_size=self.batch_size)
        # {'frame_output': (segments_num, segment_frames, classes_num),
        #   'reg_onset_output': (segments_num, segment_frames, classes_num),
        #   ...}

        audio_duration = audio_length / self.sample_rate
        frames_num = int(audio_duration * self.frames_per_second)

        # Deframe to original length.
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:frames_num]
        # {'frame_output': (N, 88*5),
        #  'reg_onset_output': (N, 88*5),
        #  ...}

        return output_dict

    def postprocess_probabilities_to_midi_events(self, output_dict):
        r"""Postprocess probabilities to MIDI events using thresholds.

        Args:
            output_dict, dict: e.g., {
                'frame_output': (N, 88*5),
                'reg_onset_output': (N, 88*5),
                ...}

        Returns:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
        """
        post_processor = RegressionPostProcessor(
            self.frames_per_second,
            classes_num=self.classes_num,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshod,
            frame_threshold=self.frame_threshold,
            pedal_offset_threshold=self.pedal_offset_threshold,
            modeling_offset=self.modeling_offset,
        )

        # programs_output_dict = {}

        '''
        for k, program in enumerate(self.programs):
            programs_output_dict[program] = {}
            for key in output_dict.keys():
                programs_output_dict[program][key] = output_dict[key][
                    :, k * self.classes_num : (k + 1) * self.classes_num
                ]
        '''
        plugins_output_dict = {}
        for k, plugin_name in enumerate(self.plugin_names):
            plugins_output_dict[plugin_name] = {}
            for key in output_dict.keys():
                plugins_output_dict[plugin_name][key] = output_dict[key][
                    :, k * self.classes_num : (k + 1) * self.classes_num
                ]
        # {'0': {
        #     'frame_output': (N, 88),
        #     'reg_onset_output': (N, 88),
        #     ...},
        #  'percussion': {
        #     'frame_output': (N, 88),
        #     'reg_onset_output': (N, 88),
        #     ...},
        #  ...}

        '''
        midi_events = {}
        for k, program in enumerate(self.programs):
            print('program: {}'.format(program))

            if program == 'percussion':
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[program],
                    detect_type='percussion',
                )


            else:
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[program],
                    detect_type='piano',
                )
            midi_events[program] = est_note_events
        '''

        '''
        midi_events = {}
        for k, plugin_name in enumerate(self.plugin_names):
            print('plugin_name: {}'.format(plugin_name))

            if plugin_name == 'percussion':
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[plugin_name],
                    detect_type='percussion',
                )


            else:
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    programs_output_dict[plugin_name],
                    detect_type='piano',
                )
            midi_events[plugin_name] = est_note_events
        '''

        midi_events = {}
        for k, plugin_name in enumerate(self.plugin_names):
            print('plugin_name: {}'.format(plugin_name))

            if plugin_name == 'percussion':
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    plugins_output_dict[plugin_name],
                    detect_type='percussion',
                )


            else:
                (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                    plugins_output_dict[plugin_name],
                    detect_type='piano',
                )
            midi_events[plugin_name] = est_note_events

        return midi_events

    '''
    def write_midi_events_to_midi_file(self, midi_events, midi_path):
        r"""Write MIDI events to a MIDI file.

        Args:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
            midi_path: str, path to write out the MIDI file

        Returns:
            None
        """
        new_midi_data = pretty_midi.PrettyMIDI()

        for k, program in enumerate(self.programs):
            if program == 'percussion':
                new_track = pretty_midi.Instrument(program=0)
                new_track.is_drum = True
            else:
                new_track = pretty_midi.Instrument(program=int(program))

            new_track.name = program
            est_note_events = midi_events[program]

            for note_event in est_note_events:
                # print(note_event['offset_time'] - note_event['onset_time'])
                if note_event['offset_time'] - note_event['onset_time'] >= 0.05:
                    new_note = pretty_midi.Note(
                        pitch=note_event['midi_note'],
                        start=note_event['onset_time'],
                        end=note_event['offset_time'],
                        velocity=note_event['velocity'],
                    )
                    new_track.notes.append(new_note)
            new_midi_data.instruments.append(new_track)

        os.makedirs(os.path.dirname(midi_path), exist_ok=True)
        new_midi_data.write(midi_path)
        print('Write out transcribed result to {}'.format(midi_path))
    '''

    def write_midi_events_to_midi_file(self, midi_events, midi_path):
        r"""Write MIDI events to a MIDI file.

        Args:
            midi_events: dict, e.g.,
                {'0': [
                    ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                    ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                    ...]
                 'percussion': [
                    ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                    ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                    ...],
                 ...}
            midi_path: str, path to write out the MIDI file

        Returns:
            None
        """
        new_midi_data = pretty_midi.PrettyMIDI()

        for k, plugin_name in enumerate(self.plugin_names):
            if plugin_name == 'percussion':
                new_track = pretty_midi.Instrument(program=0)
                new_track.is_drum = True
            else:
                instrument_name = PLUGIN_NAME_TO_INSTRUMENT[plugin_name]
                program = pretty_midi.instrument_name_to_program(instrument_name)
                new_track = pretty_midi.Instrument(program=int(program))

            new_track.name = str('{}_{}'.format(program, plugin_name))
            est_note_events = midi_events[plugin_name]

            for note_event in est_note_events:
                # print(note_event['offset_time'] - note_event['onset_time'])
                if note_event['offset_time'] - note_event['onset_time'] >= 0.05:
                    new_note = pretty_midi.Note(
                        pitch=note_event['midi_note'],
                        start=note_event['onset_time'],
                        end=note_event['offset_time'],
                        velocity=note_event['velocity'],
                    )
                    new_track.notes.append(new_note)
            new_midi_data.instruments.append(new_track)

        os.makedirs(os.path.dirname(midi_path), exist_ok=True)
        new_midi_data.write(midi_path)
        print('Write out transcribed result to {}'.format(midi_path))

    def enframe(self, x, segment_samples):
        r"""Enframe long sequence into short segments.

        Args:
            x: (1, audio_samples)
            segment_samples: int, e.g., 10 * sample_rate

        Returns:
            batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        r"""Deframe predicted segments to original sequence.

        Args:
            x: (N, segment_frames, classes_num)

        Returns:
            y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0:-1, :]
            # Remove an extra frame in the end of each segment caused by the
            # 'center=True' argument when calculating spectrogram.
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y

    def _forward_mini_batches(self, model, x, conditions, batch_size):
        r"""Forward data to model in mini-batch.

        Args:
            model: nn.Module
            x: ndarray, (N, segment_samples)
            batch_size: int

        Returns:
            output_dict: dict, e.g. {
                'frame_output': (segments_num, frames_num, classes_num),
                'onset_output': (segments_num, frames_num, classes_num),
                ...}
        """
        output_dict = {}

        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = x[pointer : pointer + batch_size]
            batch_condition = conditions[pointer : pointer + batch_size]

            if 'cuda' in self.device:
                batch_waveform = torch.Tensor(batch_waveform).to(self.device)
                batch_condition = torch.Tensor(batch_condition).to(self.device)

            pointer += batch_size

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_waveform, batch_condition)

            for key in batch_output_dict.keys():
                self._append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def _append_to_dict(self, dict, key, value):
        if key in dict.keys():
            dict[key].append(value)
        else:
            dict[key] = [value]


class OnsetFramePostProcessor(object):
    def __init__(
        self,
        frames_per_second,
        onset_threshold,
        offset_threshold,
        frame_threshold,
        pedal_offset_threshold,
        modeling_offset=True,
    ):
        r"""Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
            frames_per_second: int
            classes_num: int
            onset_threshold: float
            offset_threshold: float
            frame_threshold: float
            pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = BEGIN_NOTE
        self.velocity_scale = VELOCITY_SCALE
        self.modeling_offset = modeling_offset

    
    def output_dict_to_midi_events(self, output_dict, detect_type):
        # detect_type is a dummy variable
        """
        Finds the note timings based on the onsets and frames information
        Parameters
        ----------
        onsets: torch.FloatTensor, shape = [frames, bins]
        frames: torch.FloatTensor, shape = [frames, bins]
        velocity: torch.FloatTensor, shape = [frames, bins]
        onset_threshold: float
        frame_threshold: float
        Returns
        -------
        pitches: np.ndarray of bin_indices
        intervals: np.ndarray of rows containing (onset_index, offset_index)
        velocities: np.ndarray of velocity values
        """
        frames = (output_dict['frame_output']>self.frame_threshold)
        onsets = (output_dict['reg_onset_output']>self.onset_threshold)
        
#         onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step
        onset_diff = onsets & frames # New condition such that both onset and frame on to get a note
    
        frames = frames.float()
        onsets = onsets.float()        

#         pitches = []
#         intervals = []
        midi_events = []

        for nonzero in torch.nonzero(onset_diff, as_tuple=False):
            frame = nonzero[0].item()
            pitch = nonzero[1].item()
            
            if onset_diff[frame-1, pitch]==1: # ignore consective onsets
                continue

            onset = frame
            offset = frame

            # This while loop is looking for where does the note ends
            while frames[offset, pitch].item():
                offset += 1
                if offset == frames.shape[0]:
                    break

            # After knowing where does the note start and end, we can return the pitch information (and velocity)
            # Since we don't transcribe velocity, we will hard code it to 100
            if offset > onset:
                midi_events.append(
                    {
                        'onset_time': onset/self.frames_per_second,
                        'offset_time': offset/self.frames_per_second,
                        'midi_note': pitch+BEGIN_NOTE, # Need to offset the pianoroll lowest note
                        'velocity': 100,
                    }
                )
#                 pitches.append(pitch)
#                 intervals.append([onset/self.frames_per_second, offset/self.frames_per_second, pitch, 100])
        return midi_events, []
            
class RegressionPostProcessor(object):
    def __init__(
        self,
        frames_per_second,
        onset_threshold,
        offset_threshold,
        frame_threshold,
        pedal_offset_threshold,
        modeling_offset=True,
    ):
        r"""Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
            frames_per_second: int
            classes_num: int
            onset_threshold: float
            offset_threshold: float
            frame_threshold: float
            pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = BEGIN_NOTE
        self.velocity_scale = VELOCITY_SCALE
        self.modeling_offset = modeling_offset

    def output_dict_to_midi_events(self, output_dict, detect_type):
        r"""Main function. Post process model outputs to MIDI events.

        Args:
            output_dict: {
              'reg_onset_output': (segment_frames, classes_num),
              'reg_offset_output': (segment_frames, classes_num),
              'frame_output': (segment_frames, classes_num),
              'velocity_output': (segment_frames, classes_num),
              'reg_pedal_onset_output': (segment_frames, 1),
              'reg_pedal_offset_output': (segment_frames, 1),
              'pedal_frame_output': (segment_frames, 1)}

        Outputs:
            est_note_events: list of dict, e.g. [
                {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83},
                {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

            est_pedal_events: list of dict, e.g. [
                {'onset_time': 0.17, 'offset_time': 0.96},
                {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = self.output_dict_to_note_pedal_arrays(output_dict, detect_type)
        # est_on_off_note_vels: (events_num, 4), the four columns are:
        # [onset_time, offset_time, piano_note, velocity],
        # est_pedal_on_offs: (pedal_events_num, 2), the two columns are:
        # [onset_time, offset_time]

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)
        
        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)    

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict, detect_type):
        r"""Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
            output_dict: dict, {
                'reg_onset_output': (frames_num, classes_num),
                'reg_offset_output': (frames_num, classes_num),
                'frame_output': (frames_num, classes_num),
                'velocity_output': (frames_num, classes_num),
                ...}

        Returns:
            est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time,
                offset_time, piano_note and velocity. E.g. [
                [39.74, 39.87, 27, 0.65],
                [11.98, 12.11, 33, 0.69],
                ...]

            est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time
                and offset_time. E.g. [
                [0.17, 0.96],
                [1.17, 2.65],
                ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = self.get_binarized_output_from_regression(
            reg_output=output_dict['reg_onset_output'], threshold=self.onset_threshold, neighbour=2
        )

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output

        # Calculate binarized offset output from regression output
        if self.modeling_offset:
            (offset_output, offset_shift_output) = self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'],
                threshold=self.offset_threshold,
                neighbour=4,
            )

            output_dict['offset_output'] = offset_output  # Values are 0 or 1
            output_dict['offset_shift_output'] = offset_shift_output

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Pedal onsets are not used in inference. Instead, frame-wise pedal
            # predictions are used to detect onsets. We empirically found this is
            # more accurate to detect pedal onsets.
            pass

        if 'reg_pedal_offset_output' in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output,) = self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_pedal_offset_output'],
                threshold=self.pedal_offset_threshold,
                neighbour=4,
            )

            output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict, detect_type)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)

        else:
            est_pedal_on_offs = None

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        r"""Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
            reg_output: (frames_num, classes_num)
            threshold: float
            neighbour: int

        Returns:
            binary_output: (frames_num, classes_num)
            shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape

        activation_indices = (reg_output > threshold).nonzero(as_tuple=False)

        for idx in activation_indices:
            n = idx[0]
            k = idx[1]
            if self.is_monotonic_neighbour(reg_output, idx[0], idx[1], neighbour):
                binary_output[n, k] = 1
                # See Section III-D in [1] for deduction.
                # [1] Q. Kong, et al., High-resolution Piano Transcription
                # with Pedals by Regressing Onsets and Offsets Times, 2020.
                if n+1 == len(reg_output):
                    pass # ignore the last index
                else:
                    if reg_output[n - 1, k] > reg_output[n + 1, k]:
                        shift = (reg_output[n + 1, k] - reg_output[n - 1, k]) / (reg_output[n, k] - reg_output[n + 1, k]) / 2
                    else:
                        shift = (reg_output[n + 1, k] - reg_output[n - 1, k]) / (reg_output[n, k] - reg_output[n - 1, k]) / 2
                    shift_output[n, k] = shift    

        return binary_output, shift_output

    def is_monotonic_neighbour(self, reg_output, t_idx, f_idx, neighbour):
        r"""Detect if values are monotonic in both side of x[n].

        Args:
            x: (frames_num,)
            n: int
            neighbour: int

        Returns:
            monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if reg_output[t_idx - i, f_idx] < reg_output[t_idx - i - 1, f_idx]:
                monotonic = False
            try:
                if reg_output[t_idx + i, f_idx] < reg_output[t_idx + i + 1, f_idx]:
                    monotonic = False
            except:
                print(f"Reaching the end of the roll {t_idx + i + 1} out of {reg_output.shape}")

        return monotonic

    def output_dict_to_detected_notes(self, output_dict, detect_type='piano'):
        r"""Postprocess output_dict to piano notes.

        Args:
            output_dict: dict, e.g. {
                'onset_output': (frames_num, classes_num),
                'onset_shift_output': (frames_num, classes_num),
                'offset_output': (frames_num, classes_num),
                'offset_shift_output': (frames_num, classes_num),
                'frame_output': (frames_num, classes_num),
                'onset_output': (frames_num, classes_num),
                ...}

        Returns:
            est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets,
            MIDI notes and velocities. E.g.,
                [[39.7375, 39.7500, 27., 0.6638],
                 [11.9824, 12.5000, 33., 0.6892],
                 ...]
        """
        est_tuples = []
        est_midi_notes = []
        frames_num, classes_num = output_dict['frame_output'].shape

        for piano_note in range(classes_num):

            if 'velocity_output' in output_dict.keys():
                velocity_array = output_dict['velocity_output'][:, piano_note] * self.velocity_scale
            else:
                # Set velocity to 100 by default.
                velocity_array = np.ones(frames_num) * 100

            if detect_type == 'piano':
                # Detect piano notes.
                if self.modeling_offset:
                    est_tuples_per_note = note_detection_with_onset_offset_regress(
                        frame_output=output_dict['frame_output'][:, piano_note],
                        onset_output=output_dict['onset_output'][:, piano_note],
                        onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                        offset_output=output_dict['offset_output'][:, piano_note],
                        offset_shift_output=output_dict['offset_shift_output'][:, piano_note],
                        velocity_output=velocity_array,
                        frame_threshold=self.frame_threshold,
                    )
                else:
                    est_tuples_per_note = note_detection_with_onset_regress(
                        frame_output=output_dict['frame_output'][:, piano_note],
                        onset_output=output_dict['onset_output'][:, piano_note],
                        onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                        velocity_output=velocity_array,
                        frame_threshold=self.frame_threshold,
                    )
                    
            elif detect_type == 'percussion':
                est_tuples_per_note = drums_detection_with_onset_regress(
                    onset_output=output_dict['onset_output'][:, piano_note],
                    onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                    velocity_output=velocity_array,
                )

            else:
                raise NotImplementedError

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)
            
        if est_tuples: # check if est_tuples is empty
        
            est_tuples = np.array(est_tuples)  # (notes, 5)
            # (notes, 5), the five columns are onset, offset, onset_shift,
            # offset_shift and normalized_velocity.

            est_midi_notes = np.array(est_midi_notes)  # (notes,)

            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            velocities = est_tuples[:, 4]

            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            # (notes, 3), the three columns are onset_times, offset_times and velocity.

            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)
        else:
            print(f"empty pianoroll")
            est_on_off_note_vels = np.array([])

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        r"""Postprocess output_dict to piano pedals.

        Args:
            output_dict: dict, e.g. {
                'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
            est_on_off: (notes, 2), the two columns are pedal onsets and pedal
                offsets. E.g.,
                [[0.1800, 0.9669],
                 [1.1400, 2.6458],
                 ...]
        """
        frames_num = output_dict['pedal_frame_output'].shape[0]

        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0],
            offset_output=output_dict['pedal_offset_output'][:, 0],
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0],
            frame_threshold=0.5,
        )

        est_tuples = np.array(est_tuples)
        # (notes, 2), the two columns are pedal onsets and pedal offsets.

        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        r"""Reformat detected notes to midi events.

        Args:
            est_on_off_vels: (notes, 3), the three columns are onset_times,
                offset_times and velocity. E.g.
                [[32.8376, 35.7700, 0.7932],
                 [37.3712, 39.9300, 0.8058],
                 ...]

        Returns:
            midi_events, list, e.g.,
                [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
                 {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
                 ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append(
                {
                    'onset_time': est_on_off_note_vels[i][0],
                    'offset_time': est_on_off_note_vels[i][1],
                    'midi_note': int(est_on_off_note_vels[i][2]),
                    'velocity': int(est_on_off_note_vels[i][3]),
                }
            )

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        r"""Reformat detected pedal onset and offsets to events.

        Args:
            pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
                [[0.1800, 0.9669],
                 [1.1400, 2.6458],
                 ...]

        Returns:
            pedal_events: list of dict, e.g.,
                [{'onset_time': 0.1800, 'offset_time': 0.9669},
                 {'onset_time': 1.1400, 'offset_time': 2.6458},
                 ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({'onset_time': pedal_on_offs[i, 0], 'offset_time': pedal_on_offs[i, 1]})

        return pedal_events


def load_meta(meta_path):

    with open(filename_info, 'r') as stream:
        try:
            metadata = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def inference_filter(args):

    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    cluster_condition_path = args.cluster_condition_path
    output_midi_path = args.output_midi_path
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    # condition_type = 'hard'

    cluster_dict = pickle.load(open(cluster_condition_path, 'rb'))
    program_conditions = cluster_dict['program_condition']
    program_plugin_ids = cluster_dict['program_plugin_id']

    # clustered_plugin_ids = cluster_dict['clustered_plugin_id']

    condition_size = len(program_conditions[-1])

    plugin_names = []
    for plugin_id in program_plugin_ids:
        plugin_name = PLUGIN_IX_TO_LB[plugin_id]
        plugin_names.append(plugin_name)
        
    configs = read_yaml(config_yaml)
    model_type = configs['train']['model_type']
    modeling_offset = False
    modeling_velocity = configs['train']['modeling_velocity']

    plugin_labels_num = PLUGIN_LABELS_NUM
    sample_rate = SAMPLE_RATE
    tagging_segment_seconds = TAGGING_SEGMENT_SECONDS
    frames_per_second = FRAMES_PER_SECOND
    piano_notes_num = CLASSES_NUM

    segment_seconds = 10.
    segment_samples = int(segment_seconds * sample_rate)
    segment_frames = int(segment_seconds * frames_per_second)

    # Load audio.
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # for program in programs_list:
    #     print('program: {}, {}'.format(program, program))

    transcriber = MusicTranscriber(
        condition_size=condition_size,
        plugin_names=plugin_names,
        model_type=model_type,
        modeling_offset=modeling_offset,
        modeling_velocity=modeling_velocity,
        device=device,
        checkpoint_path=checkpoint_path,
        segment_samples=segment_samples,
    )

    transcribe_time = time.time()

    if not output_midi_path:
        output_midi_path = 'results/_zz.mid'

    transcribed_dict = transcriber.transcribe(audio, program_conditions, output_midi_path)
    print('Transcribe time: {:.3f} s'.format(time.time() - transcribe_time))

    # from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_inference = subparsers.add_parser('inference_filter')
    parser_inference.add_argument('--config_yaml', type=str, required=True)
    parser_inference.add_argument('--checkpoint_path', type=str, required=True)
    parser_inference.add_argument('--audio_path', type=str, required=True)
    parser_inference.add_argument('--midi_path', type=str, default="")
    parser_inference.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'inference_filter':
        inference_filter(args)

    else:
        raise NotImplementedError
