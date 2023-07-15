import argparse
import os
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from jointist.config import SAMPLE_RATE
from jointist.dataset_creation.create_slakh2100 import write_single_audio_to_hdf5, write_single_midi_to_hdf5


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
    dataset_root = args.dataset_root
    meta_csv_path = args.meta_csv_path
    hdf5s_dir = args.hdf5s_dir
    sample_rate = SAMPLE_RATE

    df = pd.read_csv(meta_csv_path, sep=',')
    audios_num = len(df)

    # paths
    feature_extraction_time = time.time()

    for target_split in ["train", "test", "validation"]:

        params = []

        for audio_index in range(audios_num):
            split = df['split'][audio_index]
            midi_filename = df['midi_filename'][audio_index]
            audio_filename = df['audio_filename'][audio_index]

            audio_path = os.path.join(dataset_root, audio_filename)

            hdf5_path = os.path.join(hdf5s_dir, split, "{}.h5".format(os.path.splitext(midi_filename)[0]))
            os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

            param = (audio_index, audio_path, hdf5_path, audio_filename, split, sample_rate)

            if split == target_split:
                params.append(param)

        print("------ Split: {} (Total: {} clips) ------".format(target_split, len(params)))

        # Debug by uncomment the following code.
        # write_single_audio_to_hdf5(params[0])

        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor(max_workers=None) as pool:
            pool.map(write_single_audio_to_hdf5, params)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))


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
    meta_csv_path = args.meta_csv_path
    hdf5s_dir = args.hdf5s_dir

    df = pd.read_csv(meta_csv_path, sep=',')
    audios_num = len(df)

    # paths
    feature_extraction_time = time.time()

    for target_split in ["train", "test", "validation"]:
        # for target_split in ["test"]:

        params = []

        for midi_index in range(audios_num):
            split = df['split'][midi_index]
            midi_filename = df['midi_filename'][midi_index]

            # audio_path = os.path.join(dataset_root, audio_filename)
            midi_path = os.path.join(processed_midis_dir, midi_filename)

            hdf5_path = os.path.join(hdf5s_dir, split, "{}.h5".format(os.path.splitext(midi_filename)[0]))
            os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

            # if '1_funk-groove1_138_beat_4-4_1' in midi_path:

            param = (midi_index, midi_path, hdf5_path, split)

            if split == target_split:
                params.append(param)

        print("------ Split: {} (Total: {} clips) ------".format(target_split, len(params)))

        # Debug by uncomment the following code.
        # write_single_midi_to_hdf5(params[0])

        # Pack audio files to hdf5 files in parallel.
        with ProcessPoolExecutor() as pool:
            pool.map(write_single_midi_to_hdf5, params)

        # for param in params:
        #     write_single_midi_to_hdf5(param)

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_pack_audios = subparsers.add_parser("pack_audios_to_hdf5s")
    parser_pack_audios.add_argument("--dataset_root", type=str, required=True, help="Directory of groove audios.")
    parser_pack_audios.add_argument("--meta_csv_path", type=str, required=True, help="Directory of groove audios.")
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

    parser_pack_midi_events.add_argument("--meta_csv_path", type=str, required=True, help="Directory of groove audios.")

    parser_pack_midi_events.add_argument(
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
    else:
        raise Exception("Incorrect arguments!")
