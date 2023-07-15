"""
Pre-process Groove MIDIs to merge drum parts
"""

import os
import csv
import tqdm
import argparse
import multiprocessing

import numpy as np
import pretty_midi

roland2standard = {}
roland2standard[36] = 36    # Bass Drum
roland2standard[38] = 38    # Snare
roland2standard[40] = 38
roland2standard[37] = 38
roland2standard[48] = 50    # High Tom
roland2standard[50] = 50
roland2standard[45] = 47    # Low-mid Tom
roland2standard[47] = 47
roland2standard[43] = 43    # High Floor Tom
roland2standard[58] = 43
roland2standard[46] = 46    # Open Hi-Hat
roland2standard[26] = 46
roland2standard[42] = 42    # Closed Hi-Hat
roland2standard[22] = 42
roland2standard[44] = 42    
roland2standard[49] = 49    # Crash Cymbal
roland2standard[55] = 49
roland2standard[57] = 49
roland2standard[52] = 49
roland2standard[51] = 51    # Ride Cymbal
roland2standard[59] = 51
roland2standard[53] = 51
roland2standard[54] = 42  # Tambourine -> Closed Hi-Hat
roland2standard[39] = 38  # Clap -> Snare
# roland2standard[56] = 42  # Cowbell
roland2standard[56] = 51  # Cowbell -> Ride Cymbal

"""config_1: 9 part (standard, no merge)
MIDI Number     Drum part name
36              Kick
38              Snare
42              Hi-Hat
43              Low Tom
47              Mid Tom
50              High Tom
46              Open Hi-Hat
49              Crash Cymbal
51              Ride Cymbal
"""
pitch_map_config_1 = {}
pitch_map_config_1[36] = 36  # Kick
pitch_map_config_1[38] = 38  # Snare
pitch_map_config_1[42] = 42  # Closed Hi-Hat
pitch_map_config_1[43] = 43  # Low Tom
pitch_map_config_1[46] = 46  # Open Hi-Hat
pitch_map_config_1[47] = 47  # Mid Tom
pitch_map_config_1[49] = 49  # Crash Cymbal
pitch_map_config_1[50] = 50  # High Tom
pitch_map_config_1[51] = 51  # Ride Cymbal

"""config_2: 7 part (merge high/mid/low Toms)
MIDI Number     Drum part name
36              Kick
38              Snare
42              Hi-Hat
47              Toms (low/mid/high)
46              Open Hi-Hat
49              Crash Cymbal
51              Ride Cymbal
"""
pitch_map_config_2 = {}
pitch_map_config_2[36] = 36
pitch_map_config_2[38] = 38
pitch_map_config_2[42] = 42
pitch_map_config_2[43] = 47
pitch_map_config_2[46] = 46
pitch_map_config_2[47] = 47
pitch_map_config_2[49] = 49
pitch_map_config_2[50] = 47
pitch_map_config_2[51] = 51

"""config_3: 4 part
MIDI Number   Drum part name
36            Kick
38            Snare
42            Hi-Hat
47            Others
"""
pitch_map_config_3 = {}
pitch_map_config_3[36] = 36
pitch_map_config_3[38] = 38
pitch_map_config_3[42] = 42
pitch_map_config_3[43] = 47
pitch_map_config_3[46] = 47
pitch_map_config_3[47] = 47
pitch_map_config_3[49] = 47
pitch_map_config_3[50] = 47
pitch_map_config_3[51] = 47

"""config_4: 5 part (merge high/mid/low Toms)
MIDI Number     Drum part name
36              Kick
38              Snare
42              Hi-Hat
47              Toms (low/mid/high)
49              Crash Cymbal
"""
pitch_map_config_4 = {}
pitch_map_config_4[36] = 36
pitch_map_config_4[38] = 38
pitch_map_config_4[42] = 42
pitch_map_config_4[43] = 47
pitch_map_config_4[46] = 42
pitch_map_config_4[47] = 47
pitch_map_config_4[49] = 49
pitch_map_config_4[50] = 47
pitch_map_config_4[51] = 42


def get_tempo(filename: str):
    midi_data = pretty_midi.PrettyMIDI(filename)
    tempo = midi_data.get_tempo_changes()
    if len(tempo[1]) > 1:
        raise Exception("multiple tempo detected!")
    else:
        tempo = tempo[1][0]
    return tempo


def read_midi(filename: str) -> np.array:
    """
    read a MIDI file into a note matrix

    Args:
        filename (str): the filename (full path) of the MIDI file

    Return:
        notes (np.array): the note event matrix
                          each row represents: onset(sec), offset(sec), pitch(MIDInumber), velocity
                          shape=(n, 4), where n is the number of notes
    """
    midi_data = pretty_midi.PrettyMIDI(filename)
    instrument = midi_data.instruments[0]

    n_note = len(instrument.notes)
    notes = np.zeros((n_note, 4))

    for i in range(n_note):
        note_event = instrument.notes[i]
        notes[i] = note_event.start, note_event.end, note_event.pitch, note_event.velocity

    return notes


def write_midi(filename: str, notes: np.array, bpm=120) -> None:
    """
    write a note matrix into a MIDI file

    Args:
        filename (str): filename for the output midi file
        notes (np.array):   the note event matrix
                            each row represents: onset(sec), offset(sec), pitch(MIDInumber), velocity
                            shape=(n, 4), where n is the number of notes
        bpm (int): tempo of the note matrix

    Return:
        - filename (str):   the filename (full path) of the MIDI file
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instr = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
    instr.is_drum = True

    for note in notes:
        note = pretty_midi.Note(velocity=int(note[3]), pitch=int(note[2]), start=note[0], end=note[1])
        instr.notes.append(note)

    midi.instruments.append(instr)
    midi.write(filename)


def read_all_metadata(filename: str) -> list:
    """read the csv file of groove dataset"""
    with open(filename, "r") as f:
        lines = [x.strip() for x in f.readlines()]
    lines = csv.reader(lines[1:])

    metadata_list = []
    for line in lines:
        metadata = {}
        metadata["drummer"] = line[0]
        metadata["session"] = line[1]
        metadata["id"] = line[2]
        metadata["style"] = line[3]
        metadata["bpm"] = line[4]
        metadata["beat_type"] = line[5]
        metadata["time_signature"] = line[6]
        metadata["duration"] = line[7]
        metadata["split"] = line[8]
        metadata["midi_filename"] = line[9]
        metadata["audio_filename"] = line[10]
        metadata["kit_name"] = line[11]
        metadata_list.append(metadata)

    return metadata_list


def process_midi(metadata: dict) -> None:
    """a wrapper of `write_midi()` + utility"""
    midi_filename = metadata["midi_filename"]

    filename = os.path.join(path_dataset, midi_filename)
    path_dataset_processed = metadata['path_to_save']
    filename_processed = os.path.join(path_dataset_processed, midi_filename)
    if os.path.exists(filename_processed):
        return

    os.makedirs(os.path.dirname(filename_processed), exist_ok=True)

    tempo = get_tempo(filename)
    notes = read_midi(filename)
    pitches = notes[:, 2].astype("i")
    pitches_standard = [roland2standard[x] for x in pitches]
    pitch_map_config = metadata['map_config']
    pitches_mapped = [pitch_map_config[x] for x in pitches_standard]
    notes[:, 2] = pitches_mapped

    write_midi(filename_processed, notes, bpm=tempo)


def prepare_processed_midi(path_dataset: str, path_dataset_processed: str, pitch_map_config: dict):
    """main function of this script.

    Args:
        path_dataset (str): source dataset path. one that contains drummer folders and the csv file.
        path_dataset_processed (str): output path
        pitch_map_config (dict): map config (config_1, config_2, ..)
    """
    metadata_list = read_all_metadata(os.path.join(path_dataset, "e-gmd-v1.0.0.csv"))  # list of dict

    for metadata in metadata_list:
        metadata['map_config'] = pitch_map_config.copy()  # copy the mapping config for easier multiprocessing
        metadata['path_to_save'] = path_dataset_processed  # copy the mapping config for easier multiprocessing

    # tqdm + multiprocessing --> see https://github.com/tqdm/tqdm/issues/484#issuecomment-351001534
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for _ in tqdm.tqdm(pool.imap_unordered(process_midi, metadata_list), total=len(metadata_list)):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_type', type=str, choices=['config_1', 'config_2', 'config_3', 'config_4'], required=True)
    parser.add_argument('--path_dataset', type=str, required=True)
    parser.add_argument('--path_dataset_processed', type=str, required=True, help='folder to write the result')

    args = parser.parse_args()
    config_type = args.config_type
    path_dataset = args.path_dataset
    path_dataset_processed = args.path_dataset_processed

    if config_type == "config_1":
        map_config = pitch_map_config_1
    elif config_type == "config_2":
        map_config = pitch_map_config_2
    elif config_type == "config_3":
        map_config = pitch_map_config_3
    elif config_type == "config_4":
        map_config = pitch_map_config_4
    else:
        raise NotImplementedError('config type %s is not implemented yet' % config_type)

    prepare_processed_midi(path_dataset, path_dataset_processed, map_config)
