import argparse
import os
import pathlib
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import h5py
import librosa
import numpy as np
import musdb
from mido import MidiFile

from jointist.config import SAMPLE_RATE
from jointist.utils import float32_to_int16

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
    source_type = args.source_type
    hdf5s_dir = args.hdf5s_dir
    sample_rate = SAMPLE_RATE

    mono = True
    resample_type = "kaiser_fast"

    # paths
    feature_extraction_time = time.time()

    for subset in ['train', 'test']:

        mus = musdb.DB(root=dataset_root, subsets=subset)

        print("------ Split: {} (Total: {} clips) ------".format(subset, len(mus)))

        for track_index, track in enumerate(mus.tracks):

            hdf5_path = os.path.join(hdf5s_dir, subset, "{}.h5".format(track.name))
            os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

            with h5py.File(hdf5_path, "w") as hf:

                hf.attrs.create("audio_name", data=track.name.encode(), dtype="S100")
                hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)
                hf.attrs.create("split", data=subset.encode(), dtype="S20")
                # hf.attrs.create("duration", data=duration, dtype=np.float32)

                audio = track.targets[source_type].audio.T
                # (channels_num, audio_samples)

                # Preprocess audio to mono / stereo, and resample.
                audio = preprocess_audio(audio, mono, track.rate, sample_rate, resample_type)
                # (audio_samples,)

                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)

                hf.attrs.create("duration", data=len(audio) / sample_rate, dtype=np.float32)

            print("{} Write to {}, {}".format(track_index, hdf5_path, audio.shape))

    print("Time: {:.3f} s".format(time.time() - feature_extraction_time))


def preprocess_audio(audio, mono, origin_sr, sr, resample_type):
    r"""Preprocess audio to mono / stereo, and resample.

    Args:
        audio: (channels_num, audio_samples), input audio
        mono: bool
        origin_sr: float, original sample rate
        sr: float, target sample rate
        resample_type: str, e.g., 'kaiser_fast'

    Returns:
        output: ndarray, output audio
    """

    if mono:
        audio = np.mean(audio, axis=0)
        # (audio_samples,)

    output = librosa.core.resample(audio, orig_sr=origin_sr, target_sr=sr, res_type=resample_type)
    # (channels_num, audio_samples) | (audio_samples,)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_pack_audios = subparsers.add_parser("pack_audios_to_hdf5s")
    parser_pack_audios.add_argument("--dataset_root", type=str, required=True, help="Directory of Slakh2100 audios.")
    parser_pack_audios.add_argument("--source_type", type=str, required=True, help="Directory of Slakh2100 audios.")
    parser_pack_audios.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.mode == "pack_audios_to_hdf5s":
        pack_audios_to_hdf5s(args)

    else:
        raise Exception("Incorrect arguments!")
