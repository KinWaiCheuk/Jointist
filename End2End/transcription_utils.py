import numpy as np
import torch
import time
import pretty_midi
import os
from End2End.piano_vad import (
    note_detection_with_onset_offset_regress,
    note_detection_with_onset_regress,
    pedal_detection_with_onset_offset_regress,
    drums_detection_with_onset_regress
)
import sys
from End2End.constants import (
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

from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )

sample_rate = SAMPLE_RATE

MIDIclass2pretty_idx = {
    "Piano": 0,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Clavinet": 7,
    "Chromatic Percussion": 12,
    "Organ": 19,
    "Accordion": 21,
    "Harmonica": 22,
    "Acoustic Guitar": 24,
    "Electric Guitar": 27,
    "Bass": 38,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Contrabass": 43,
    "Strings": 48,
    "Harp": 46,
    "Timpani": 47,
    "Voice": 52,
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "Horn": 60,
    "Brass": 61,
    "Saxophone": 64,
    "Oboe": 68,
    "Bassoon": 70,
    "Clarinet": 71,
    "Piccolo": 72,
    "Flute": 73,
    "Recorder": 74,
    "Pipe": 75,
    "Synth Lead": 80,
    "Synth Pad": 88,
    "Synth Effects": 96,
    "Ethnic": 104,
    "Percussive": 112,
    "Sound Effects": 120
    }

def postprocess_probabilities_to_midi_events(output_dict, plugin_ids, IX_TO_NAME, classes_num, post_processor):
    # TODO refactor classes_num, post_processor
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
    start = time.time()
    plugins_output_dict = {}
    for k, plugin_name in enumerate(plugin_ids):
        plugin_name = IX_TO_NAME[plugin_name.item()]
        plugins_output_dict[plugin_name] = {}
        for key in output_dict.keys():
            plugins_output_dict[plugin_name][key] = output_dict[key][
                :, k * classes_num : (k + 1) * classes_num
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
    for k, plugin_name in enumerate(plugin_ids):
        plugin_name = IX_TO_NAME[plugin_name.item()]        
#         print('Processing plugin_name: {}'.format(plugin_name), end='\r')

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


def write_midi_events_to_midi_file(midi_events, midi_path, instrument_type):
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

    for k, plugin_name in enumerate(midi_events.keys()):
        
        if plugin_name == 'percussion':
            program = 0
            new_track = pretty_midi.Instrument(program=program)
            new_track.is_drum = True
        else:
            if instrument_type=="plugin_names":
                instrument_name = PLUGIN_NAME_TO_INSTRUMENT[plugin_name]
            elif instrument_type=="MIDI_programs":
                instrument_name = plugin_name
            elif instrument_type=="MIDI_class":
                instrument_name = plugin_name                
            else:
                raise ValueError(f"Unknow instrument type: {instrument_type}")
#             program = pretty_midi.instrument_name_to_program(instrument_name)
            if instrument_name=='Drums':
                program = 0
                new_track = pretty_midi.Instrument(program=program, is_drum=True)
            else:
                program = MIDIclass2pretty_idx[instrument_name]
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
#     print('Write out transcribed result to {}'.format(midi_path))

    
def predict_probabilities(model, audio, condition, segment_samples, segment_batch_size):
    # TODO, refactor segment_samples, model, segment_batch_size
    
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
    pad_len = int(np.ceil(audio_length / segment_samples)) * segment_samples - audio_length

#     audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
    audio = torch.cat((audio, torch.zeros((1, pad_len), device=audio.device)), axis=1)

    # Enframe to segments.
#     segments = enframe(audio, segment_samples)
    segments = audio.unfold(1, segment_samples, segment_samples//2).squeeze(0) # faster version of enframe
    # (N, segment_samples)

#     conditions = np.tile(condition, (len(segments), 1))
    conditions = condition.unsqueeze(0).repeat(len(segments),1).to(audio.device)
    
    # Inference on segments.
#     output_dict = _forward_mini_batches(model, segments, conditions, batch_size=batch_size)
    output_dict = _forward_mini_batches_torch(model, segments, conditions, batch_size=segment_batch_size)
    # {'frame_output': (segments_num, segment_frames, classes_num),
    #   'reg_onset_output': (segments_num, segment_frames, classes_num),
    #   ...}

    audio_duration = audio_length / sample_rate
    frames_num = int(audio_duration * model.frames_per_second)

    # Deframe to original length.
    for key in output_dict.keys():
        X = output_dict[key][:,:-1]   
        output_dict[key] = torch.cat((X[0,:750], X[1:-1,250:750].flatten(0,1), X[-1,250:]),0)[:frames_num].cpu()  # faster version of deframe
#         output_dict[key] = deframe(output_dict[key])[0:frames_num]
#         print(np.allclose(output_dict[key], debug.numpy()))   

    # {'frame_output': (N, 88*5),
    #  'reg_onset_output': (N, 88*5),
    #  ...}

    return output_dict



def _forward_mini_batches_torch(model, x, conditions, batch_size):
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

    model.eval()
    with torch.no_grad():
        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = x[pointer : pointer + batch_size]
            batch_condition = conditions[pointer : pointer + batch_size]

            pointer += batch_size
         
            batch_output_dict = model(batch_waveform, batch_condition)

            for key in batch_output_dict.keys():
                _append_to_dict(output_dict, key, batch_output_dict[key])

        for key in output_dict.keys():
            output_dict[key] = torch.cat(output_dict[key], axis=0)

    return output_dict

def predict_probabilities_baseline(model, audio, segment_samples, segment_batch_size):
    # TODO, refactor segment_samples, model, segment_batch_size
    
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
    pad_len = int(np.ceil(audio_length / segment_samples)) * segment_samples - audio_length

#     audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
    audio = torch.cat((audio, torch.zeros((1, pad_len), device=audio.device)), axis=1)

    # Enframe to segments.
#     segments = enframe(audio, segment_samples)
    segments = audio.unfold(1, segment_samples, segment_samples//2).squeeze(0) # faster version of enframe
    # (N, segment_samples)

    
    # Inference on segments.
#     output_dict = _forward_mini_batches(model, segments, conditions, batch_size=batch_size)
    output_dict = _forward_mini_batches_torch_baseline(model, segments, batch_size=segment_batch_size)
    # {'frame_output': (segments_num, segment_frames, classes_num),
    #   'reg_onset_output': (segments_num, segment_frames, classes_num),
    #   ...}

    audio_duration = audio_length / sample_rate
    frames_num = int(audio_duration * model.frames_per_second)

    # Deframe to original length.
    for key in output_dict.keys():
        X = output_dict[key][:,:,:-1]   # [Segments, num_instrument_classes, T, F]
        X = X.transpose(0,1) # [num_instrument_classes, Segments, T, F]
        output_dict[key] = torch.cat((X[:,0,:750], X[:,1:-1,250:750].flatten(1,2), X[:,-1,250:]),1)[:frames_num].cpu()  # faster version of deframe
#         output_dict[key] = deframe(output_dict[key])[0:frames_num]
#         print(np.allclose(output_dict[key], debug.numpy()))   

    # {'frame_output': (N, 88*5),
    #  'reg_onset_output': (N, 88*5),
    #  ...}

    return output_dict



def _forward_mini_batches_torch_baseline(model, x, batch_size):
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

    model.eval()
    with torch.no_grad():
        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = x[pointer : pointer + batch_size]

            pointer += batch_size
         
            batch_output_dict = model(batch_waveform)

            for key in batch_output_dict.keys():
                _append_to_dict(output_dict, key, batch_output_dict[key])

        for key in output_dict.keys():
            output_dict[key] = torch.cat(output_dict[key], axis=0)

    return output_dict


def _append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
        
        
def notes_to_frames(pitches, intervals, frames_per_second, shape=None):
    """
    Takes lists specifying notes sequences and return
    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    if shape==None:
        frame_num = int(np.max(intervals)*frames_per_second)
        bin_num = 128
        shape=(frame_num, bin_num)
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        onset = int(onset*frames_per_second)
        offset = int(offset*frames_per_second)
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return roll        