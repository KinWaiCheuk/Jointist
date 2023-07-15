import os
import numpy as np
import glob
import yaml
import soundfile as sf
import argparse
import pandas as pd
import time
import pathlib
import h5py
from concurrent.futures import ProcessPoolExecutor

import pretty_midi
from jointist.dataset_creation.create_slakh2100 import _parse_midi_to_events


def group_midi_tracks_plugin(
    args
):
    r"""Merge MIDI tracks into fewer closed tracks. The merging is depend on
    plugin names.

    Args:
        path_dataset: str, the path of the original dataset
        path_dataset_processed: str, the path of the processed dataset to write out
        plugin_to_closed_set: dict, a dict to map plugin name to target MIDI
            program number, e.g., {'AGML2': 0, 'bassoon': 0, ...}.

    Returns:
        None
    """
    path_dataset = args.path_dataset
    path_dataset_processed = args.path_dataset_processed

    for split in ["train", "validation", "test"]:

        path_dataset_split = os.path.join(path_dataset, split)
        path_dataset_processed_split = os.path.join(path_dataset_processed, split)
        piecenames = os.listdir(path_dataset_split)
        piecenames = [x for x in piecenames if x[0] != "."]
        piecenames.sort()
        print("total piece number in %s set: %d" % (split, len(piecenames)))

        path_midi_out = os.path.join(path_dataset_processed_split, "midi")

        os.makedirs(path_midi_out, exist_ok=True)

        for piecename in piecenames:
            print(piecename)

            # Read metadata of an audio piece. The metadata includes plugin
            # names for all tracks.
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

            # placeholder for the processed MIDI
            # filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", tracknames[0] + ".mid")
            # closed_set_program_numbers = list(plugin_to_closed_set.values())
            # closed_set_program_numbers = [int(x) for x in closed_set_program_numbers]
            # closed_set_program_numbers = np.array(closed_set_program_numbers)
            # closed_set_program_numbers = closed_set_program_numbers[
            #     closed_set_program_numbers > -1
            # ]  # remove the -1 for placeholder

            # closed_set_program_numbers = set(closed_set_program_numbers)
            # # E.g., {0, 16, 30, 33, 48}

            # midi_data_new = pretty_midi.PrettyMIDI(filename_midi)
            # midi_data_new.instruments = []

            # Program number for the processed MIDI. E.g., {0, 16, 30, 33, 48} for piano, pizz, guitar, bass, drums.
            # for program_number in closed_set_program_numbers:
            #     midi_data_new.instruments.append(pretty_midi.Instrument(program=program_number))

            # Then add drum track, as always the last track.
            # midi_data_new.instruments.append(pretty_midi.Instrument(program=0))
            # midi_data_new.instruments[-1].is_drum = True

            # Merge multiple tracks into closed set tracks.
            for trackname in tracknames:
                # E.g., "S00".

                plugin_name = metadata["stems"][trackname]["plugin_name"]

                plugin_name = os.path.splitext(os.path.basename(plugin_name))[0]
                # E.g., 'elektrik_guitar'.

                is_drum = metadata["stems"][trackname]["is_drum"]
                program_num = metadata["stems"][trackname]["program_num"]

                # Read MIDI file of a track
                filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", trackname + ".mid")
                midi_data = pretty_midi.PrettyMIDI(filename_midi)



                if len(midi_data.instruments) > 1:
                    raise Exception("multi-track midi")

                instr = midi_data.instruments[0]
                # print(instr.program, program_num)
                if not is_drum:
                    assert instr.program == program_num

                # new_program_number = int(plugin_to_closed_set[plugin_name])
                # new_program_number = 0

                # for instr_new in midi_data_new.instruments[:-1]:
                    
                #     if instr_new.program == new_program_number:

                new_midi_data = pretty_midi.PrettyMIDI()
                new_track = pretty_midi.Instrument(program=0)

                for note in instr.notes:
                    if program_num in range(32, 40):
                        note.pitch -= 12
                    new_track.notes.append(note)

                new_midi_data.instruments.append(new_track)

                out_midi_path = os.path.join(path_dataset_processed, split, piecename, '{}.mid'.format(trackname))
                os.makedirs(os.path.dirname(out_midi_path), exist_ok=True)
                new_midi_data.write(out_midi_path)
            # from IPython import embed; embed(using=False); os._exit(0)

            # filename_midi_out = os.path.join(path_midi_out, piecename + ".mid")
            # midi_data_new.write(filename_midi_out)


def pack_midis_into_hdf5s(args):
    # arguments & parameters
    processed_midis_dir = args.processed_midis_dir
    hdf5s_dir = args.hdf5s_dir

    feature_extraction_time = time.time()

    for split in ["train", "test", "validation"]:
        split_midis_dir = os.path.join(processed_midis_dir, split)
        split_hdf5s_dir = os.path.join(hdf5s_dir, split)
        os.makedirs(split_hdf5s_dir, exist_ok=True)
        # from IPython import embed; embed(using=False); os._exit(0)

        midi_names = sorted(os.listdir(split_midis_dir))

        print("------ Split: {} (Total: {} files) ------".format(split, len(midi_names)))

        params = []
        for midi_index, midi_name in enumerate(midi_names):
            midis_dir = os.path.join(split_midis_dir, midi_name)
            # hdf5_path = os.path.join(split_hdf5s_dir, "{}.h5".format(pathlib.Path(midi_path).stem))
            _hdf5s_dir = os.path.join(split_hdf5s_dir, midi_name)

            # param = (midi_index, midi_path, hdf5_path, split)
            param = (midi_index, midis_dir, _hdf5s_dir, split)
            # E.g, (0, './workspaces/processed_midi_files/closed_set_config_1/train/midi/Track00001.mid',
            # './workspaces/dataset-slakh2100/hdf5s/midi_events/closed_set_config_1/train/Track00001.h5', 'train')

            params.append(param)

        # Debug by uncomment the following code.
        # write_single_midi_to_hdf5(params[0])

        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor() as pool:
            pool.map(write_single_midi_to_hdf5, params)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))


def write_single_midi_to_hdf5(param):
    r"""Write the MIDI events of a single MIDI file into an hdf5 file.

    Args:
        param: (midi_index, midi_path, hdf5_path, split)

    Returns:
        None
    """

    [n, midis_dir, hdf5s_dir, split] = param

    midi_names = sorted(os.listdir(midis_dir))

    for midi_name in midi_names:

        midi_path = os.path.join(midis_dir, midi_name)

        midi_dict = _parse_midi_to_events(midi_path)

        # audio_name = pathlib.Path(midi_path).stem
        os.makedirs(hdf5s_dir, exist_ok=True)

        hdf5_path = os.path.join(hdf5s_dir, '{}.h5'.format(pathlib.Path(midi_name).stem))

        with h5py.File(hdf5_path, "w") as hf:
            # hf.attrs.create("audio_name", data=audio_name.encode(), dtype="S100")
            # hf.attrs.create("split", data=split.encode(), dtype="S20")

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_pack_audios = subparsers.add_parser("create_midis")
    parser_pack_audios.add_argument('--path_dataset', type=str, required=True)
    parser_pack_audios.add_argument('--path_dataset_processed', type=str, required=True)

    
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

    args = parser.parse_args()
    
    if args.mode == 'create_midis':
        group_midi_tracks_plugin(args)
    elif args.mode == 'pack_midi_events_to_hdf5s':
        pack_midis_into_hdf5s(args)