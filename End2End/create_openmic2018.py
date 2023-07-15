import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import pretty_midi
import yaml
import glob
import h5py
import librosa
import pickle
import numpy as np
from mido import MidiFile
import sys
sys.path.insert(0, "./")
from End2End.constants import SAMPLE_RATE
from End2End.utils import float32_to_int16
# from create_slakh2100 import load_midi_track_group_info_plugin


def pack_audios_to_hdf5s(args):
    r"""Load & resample audios of the Slakh2100 dataset, then write them into
    hdf5 files.

    Args:
        dataset_dir: str, directory of dataset
        workspace: str, directory of your workspace

    Returns:
        None
    """

    # arguments & parameters
    audios_dir = args.audios_dir
    hdf5s_dir = args.hdf5s_dir
    sample_rate = SAMPLE_RATE

    # paths
    feature_extraction_time = time.time()
    
    

    for split in ["train", "test"]:
        with open(os.path.join(audios_dir, 'partitions', f"split01_{split}.csv")) as f:
            filenames = f.read().splitlines()
                      
        
        split_hdf5s_dir = os.path.join(hdf5s_dir, split)
        os.makedirs(split_hdf5s_dir, exist_ok=True)

        split_audios_dir = os.path.join(audios_dir, split) # Not needed for this dataset

        print("------ Split: {} (Total: {} clips) ------".format(split, len(filenames)))

        params = []
        for audio_index, name in enumerate(filenames):
            folder = name[:3]
            audio_name = name + '.ogg'
            audio_path = os.path.join(audios_dir, 'audio', folder, audio_name)
            
#             audio_path = os.path.join(split_audios_dir, audio_name, "mix.flac")
            hdf5_path = os.path.join(split_hdf5s_dir, "{}.h5".format(name))            
            
            param = (audio_index, audio_path, hdf5_path, audio_name, split, sample_rate)
            # E.g., (0, './datasets/dataset-slakh2100/slakh2100_flac/train/Track00001/mix.flac',
            # './workspaces/hdf5s/waveforms/train/Track00001.h5', 'Track00001', 'train', 16000)

            params.append(param)            

        # Debug by uncomment the following code.
        # write_single_audio_to_hdf5(params[0])
        
        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor(max_workers=None) as pool:
            pool.map(write_single_audio_to_hdf5, params)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))


def write_single_audio_to_hdf5(param):
    r"""Write a single audio file into an hdf5 file.

    Args:
        param: (audio_index, audio_path, hdf5_path, audio_name, split, sample_rate)

    Returns:
        None
    """

    [n, audio_path, hdf5_path, audio_name, split, sample_rate] = param
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs.create("audio_name", data=audio_name.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)
        hf.attrs.create("split", data=split.encode(), dtype="S20")
        hf.attrs.create("duration", data=duration, dtype=np.float32)
        hf.create_dataset(name="waveform", data=float32_to_int16(audio), dtype=np.int16)

    print("{} Write to {}".format(n, hdf5_path))

'''
def pack_midi_events_to_hdf5s(args):
    r"""Extract MIDI events of the processed Slakh2100 dataset, and write the
    MIDI events to hdf5 files. The processed MIDI files are obtained by merging
    tracks from open set tracks to predefined tracks, such as `piano`, `drums`,
    `strings`, etc.

    Args:
        processed_midis_dir: str, directory of processed MIDI files
        hdf5s_dir: str, directory to write out hdf5 files

    Returns:
        None
    """

    # arguments & parameters
    processed_midis_dir = args.processed_midis_dir
    hdf5s_dir = args.hdf5s_dir

    feature_extraction_time = time.time()

    for split in ["train", "test", "validation"]:
        split_midis_dir = os.path.join(processed_midis_dir, split, "midi")
        split_hdf5s_dir = os.path.join(hdf5s_dir, split)
        os.makedirs(split_hdf5s_dir, exist_ok=True)

        print("------ Split: {} (Total: {} files) ------".format(split, len(split_midis_dir)))

        midi_names = sorted(os.listdir(split_midis_dir))

        params = []
        for midi_index, midi_name in enumerate(midi_names):
            midi_path = os.path.join(split_midis_dir, midi_name)
            hdf5_path = os.path.join(split_hdf5s_dir, "{}.h5".format(pathlib.Path(midi_path).stem))

            param = (midi_index, midi_path, hdf5_path, split)
            # E.g, (0, './workspaces/processed_midi_files/closed_set_config_1/train/midi/Track00001.mid',
            # './workspaces/dataset-slakh2100/hdf5s/midi_events/closed_set_config_1/train/Track00001.h5', 'train')

            params.append(param)

        # Debug by uncomment the following code.
        # write_single_midi_to_hdf5(params[0])

        # for param in params:
        #     write_single_midi_to_hdf5(param)
        # asdf

        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor() as pool:
            pool.map(write_single_midi_to_hdf5, params)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))
'''

def pack_midi_events_to_hdf5s(args):
    r"""Extract MIDI events of the processed Slakh2100 dataset, and write the
    MIDI events to hdf5 files. The processed MIDI files are obtained by merging
    tracks from open set tracks to predefined tracks, such as `piano`, `drums`,
    `strings`, etc.

    Args:
        processed_midis_dir: str, directory of processed MIDI files
        hdf5s_dir: str, directory to write out hdf5 files

    Returns:
        None
    """

    # arguments & parameters
    processed_midis_dir = args.processed_midis_dir
    hdf5s_dir = args.hdf5s_dir

    feature_extraction_time = time.time()

    for split in ["train", "test", "validation"]:
        split_midis_dir = os.path.join(processed_midis_dir, split, "midi")
        split_hdf5s_dir = os.path.join(hdf5s_dir, split)
        os.makedirs(split_hdf5s_dir, exist_ok=True)

        print("------ Split: {} (Total: {} files) ------".format(split, len(split_midis_dir)))

        midi_names = sorted(os.listdir(split_midis_dir))

        params = []
        for midi_index, midi_name in enumerate(midi_names):
            midi_path = os.path.join(split_midis_dir, midi_name)
            hdf5_path = os.path.join(split_hdf5s_dir, "{}.pkl".format(pathlib.Path(midi_path).stem))

            param = (midi_index, midi_path, hdf5_path, split)
            # E.g, (0, './workspaces/processed_midi_files/closed_set_config_1/train/midi/Track00001.mid',
            # './workspaces/dataset-slakh2100/hdf5s/midi_events/closed_set_config_1/train/Track00001.h5', 'train')

            params.append(param)

        # Debug by uncomment the following code.
        # write_single_midi_to_hdf5_pretty_midi(params[0])
        # asdf

        # for param in params:
        #     write_single_midi_to_hdf5_pretty_midi(param)
        # asdf

        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor() as pool:
            pool.map(write_single_midi_to_hdf5_pretty_midi, params)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))

def write_single_midi_to_hdf5(param):
    r"""Write the MIDI events of a single MIDI file into an hdf5 file.

    Args:
        param: (midi_index, midi_path, hdf5_path, split)

    Returns:
        None
    """
    [n, midi_path, hdf5_path, split] = param

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    print(len(midi_data.get_tempo_changes()[0]))
    assert len(midi_data.get_tempo_changes()[0]) == 1, "Tempo must be constant"

    midi_dict = _parse_midi_to_events(midi_path)

    audio_name = pathlib.Path(midi_path).stem

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs.create("audio_name", data=audio_name.encode(), dtype="S100")
        hf.attrs.create("split", data=split.encode(), dtype="S20")

        for program in midi_dict.keys():
            hf.create_group(program.encode())

            hf[str(program).encode()].create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict[program]["midi_event"]],
                dtype="S100",
            )

            hf[str(program).encode()].create_dataset(
                name="midi_event_time",
                data=midi_dict[program]["midi_event_time"],
                dtype=np.float32,
            )

    print("{} Write hdf5 to {}".format(n, hdf5_path))


def write_single_midi_to_hdf5_pretty_midi(param):
    r"""Write the MIDI events of a single MIDI file into an hdf5 file.

    Args:
        param: (midi_index, midi_path, hdf5_path, split)

    Returns:
        None
    """
    [n, midi_path, hdf5_path, split] = param

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    midi_data.instruments[0].notes
    # from IPython import embed; embed(using=False); os._exit(0)
    # print(len(midi_data.get_tempo_changes()[0]))
    # assert len(midi_data.get_tempo_changes()[0]) == 1, "Tempo must be constant"

    # midi_dict = _parse_midi_to_events(midi_path)

    audio_name = pathlib.Path(midi_path).stem

    events_dict = {}

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            program = "percussion"
        else:
            program = instrument.program

        note_events = []

        for note in instrument.notes:
            pitch = note.pitch

            if instrument.program in range(32, 40):
                pitch -= 12

            note_event = {
                'start': note.start,
                'end': note.end,
                'pitch': pitch,
                'velocity': note.velocity,
                }
            note_events.append(note_event)
        
        events_dict[str(program)] = {
            'audio_name': audio_name,
            'program_num': instrument.program,
            'note_event': note_events,
        }

    events_dict['beats'] = midi_data.get_beats()
    events_dict['downbeats'] = midi_data.get_downbeats()
    
    pickle.dump(events_dict, open(hdf5_path, 'wb'))
    print('Write out to {}'.format(hdf5_path))


    print("{} Write hdf5 to {}".format(n, hdf5_path))


def _parse_midi_to_events(midi_path):
    r"""Parse a MIDI file into MIDI events.

    Args:
        midi_path: str

    Returns:
        midi_dict: dict, e.g. {
            '0': {'midi_event': [
                      'program_change channel=0 program=0 time=0',
                      'control_change channel=0 control=64 value=127 time=0',
                      'control_change channel=0 control=64 value=63 time=236',
                      ...],
                  'midi_event_time': [0., 0, 0.98307292, ...]}
            '16': ...,
            'percussion': ...,
            }
    """

    midi_file = MidiFile(midi_path)

    ticks_per_beat = midi_file.ticks_per_beat
    # Tick is the quantized step in a beat, e.g., ticks_per_beat = 96 indicates
    # there are 96 ticks in a beat.

    # The first track contains tempo. MIDI format uses microseconds_per_beat to
    # denote tempo.
    microseconds_per_beat = midi_file.tracks[0][0].tempo
    # E.g., microseconds_per_beat = 500000 indicates bps = 2, bpm = 120.

    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    midi_dict = {}

    # The second to the last tracks contain notes information of instruments.
    for k, track in enumerate(midi_file.tracks[1:]):
        message_list = []
        ticks = 0
        time_in_second = []

        for message in track:
            message_list.append(str(message))
            ticks += message.time
            time_in_second.append(ticks / ticks_per_second)

        if track[0].channel == 9:
            program = "percussion"
        else:
            program = track[0].program
            # program is an integer ranging from 0 to 127.

        midi_dict[str(program)] = {
            "midi_event": np.array(message_list),
            "midi_event_time": np.array(time_in_second),
        }

    return midi_dict


def pack_per_track_midi_events_to_hdf5s(args):

    dataset_dir = args.dataset_dir
    hdf5s_dir = args.hdf5s_dir

    # plugin_to_closed_set = load_midi_track_group_info_plugin(config_csv_path)

    for split in ["train", "validation", "test"]:

        path_dataset_split = os.path.join(dataset_dir, split)
        # path_dataset_processed_split = os.path.join(path_dataset_processed, split)
        piecenames = os.listdir(path_dataset_split)
        piecenames = [x for x in piecenames if x[0] != "."]
        piecenames.sort()
        print("total piece number in %s set: %d" % (split, len(piecenames)))

        # path_midi_out = os.path.join(path_dataset_processed_split, "midi")

        # os.makedirs(path_midi_out, exist_ok=True)

        for n, piecename in enumerate(piecenames):
            print(n, piecename)
            split_hdf5s_dir = os.path.join(hdf5s_dir, split)
            os.makedirs(split_hdf5s_dir, exist_ok=True) 
            hdf5_path = os.path.join(split_hdf5s_dir, "{}.pkl".format(piecename))

            filename_info = os.path.join(path_dataset_split, piecename, "metadata.yaml")

            with open(filename_info, 'r') as stream:
                try:
                    metadata = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # Get the trackname.
            path_midi = os.path.join(path_dataset_split, piecename, "MIDI")
            tracknames = glob.glob(os.path.join(path_midi, "*.mid"))
            tracknames = [os.path.splitext(os.path.basename(x))[0] for x in tracknames]
            tracknames.sort()  # ["S01", "S02", "S03", ...]
            print('Tracks num: {}'.format(len(tracknames)))

            events_dict = {}

            # Merge multiple tracks into closed set tracks.
            for trackname in tracknames:
                # E.g., "S00".

                plugin_name = metadata["stems"][trackname]["plugin_name"]
                is_drum = metadata["stems"][trackname]["is_drum"]
                program_num = metadata["stems"][trackname]["program_num"]

                if not is_drum:

                    plugin_name = os.path.splitext(os.path.basename(plugin_name))[0]
                    # E.g., 'elektrik_guitar'.

                    # Read MIDI file of a track
                    midi_path = os.path.join(path_dataset_split, piecename, "MIDI", trackname + ".mid")
                    # midi_dict = _parse_midi_to_events2(midi_path)

                    midi_data = pretty_midi.PrettyMIDI(midi_path)

                    if len(midi_data.instruments) > 1:
                        raise Exception("multi-track midi")

                    instr = midi_data.instruments[0]

                    note_events = []

                    for note in instr.notes:
                        pitch = note.pitch

                        if program_num in range(32, 40):
                            pitch -= 12

                        note_event = {
                            'start': note.start,
                            'end': note.end,
                            'pitch': pitch,
                            'velocity': note.velocity,
                            }
                        note_events.append(note_event)
                        
                    events_dict[trackname] = {
                        'audio_name': piecename,
                        'program_num': program_num,
                        'plugin_name': plugin_name,
                        'note_event': note_events,
                    }

            pickle.dump(events_dict, open(hdf5_path, 'wb'))
            print('Write out to {}'.format(hdf5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_pack_audios = subparsers.add_parser("pack_audios_to_hdf5s")
    parser_pack_audios.add_argument("--audios_dir", type=str, required=True, help="Directory of Slakh2100 audios.")
    parser_pack_audios.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )

    parser_pack_midi_events = subparsers.add_parser("pack_midi_events_to_hdf5s")
    parser_pack_midi_events.add_argument(
        "--processed_midis_dir",
        type=str,
        required=True,
        help="Directory of processed MIDI files.",
    )
    parser_pack_midi_events.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )

    parser_pack_per_track_midi_events = subparsers.add_parser("pack_per_track_midi_events_to_hdf5s")
    parser_pack_per_track_midi_events.add_argument('--dataset_dir', type=str, required=True)
    parser_pack_per_track_midi_events.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )


    # Parse arguments
    args = parser.parse_args()

    if args.mode == "pack_audios_to_hdf5s":
        pack_audios_to_hdf5s(args)
    elif args.mode == "pack_midi_events_to_hdf5s":
        pack_midi_events_to_hdf5s(args)
    elif args.mode == "pack_per_track_midi_events_to_hdf5s":
        pack_per_track_midi_events_to_hdf5s(args)
    else:
        raise Exception("Incorrect arguments!")

