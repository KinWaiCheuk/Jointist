"""Run this file after running `unzip_vocals_only.py` in mixing-secrets repo
"""
import os
import tqdm
import argparse
import tempfile
import multiprocessing

import numpy as np
import auditok
import librosa
import soundfile as sf

TARGET_SR = 16000


def _process_one(args):
    """load audio, resample and save it again to a tempo dir, split it."""
    source_path, target_path, wav_fn = args
    with tempfile.TemporaryDirectory() as temp_dir:
        src, sr = librosa.load(os.path.join(source_path, wav_fn), sr=TARGET_SR, mono=True, dtype=np.float32)

        src = np.expand_dims(src, 1)  # time. ch for pysoundfile

        resampled_audio_path = os.path.join(temp_dir, 'audio.wav')
        sf.write(resampled_audio_path, src, sr, subtype='PCM_16')

        audio_regions = auditok.split(
            resampled_audio_path,
            max_dur=10.0,
            max_silence=1.0,
            sampling_rate=sr,
            channels=1,
            sample_width=1,
            drop_trailing_silence=True,
            analysis_window=1.0,
        )
        for i, r in enumerate(audio_regions):
            filename = r.save(
                os.path.join(
                    target_path, wav_fn.replace(' ', '-').replace('.wav', '') + "_{meta.start:.3f}-{meta.end:.3f}.wav"
                )
            )
            # print("region saved as: {}".format(filename))


def main(source_path, target_path):
    os.makedirs(target_path, exist_ok=True)

    wav_fns = [f for f in os.listdir(source_path) if f.endswith('wav')]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for _ in tqdm.tqdm(
        pool.imap_unordered(_process_one, [(source_path, target_path, wav_fn) for wav_fn in wav_fns]),
        total=len(wav_fns),
    ):
        pass

    pool.close()
    pool.join()
    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help='path that has all vocal stems')
    parser.add_argument('--target_path', type=str, required=True, help='path to save segmented vocal stems')

    args = parser.parse_args()

    main(args.source_path, args.target_path)
