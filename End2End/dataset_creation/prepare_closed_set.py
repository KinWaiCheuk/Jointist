import os
import numpy as np
import glob
import yaml
import soundfile as sf
import argparse
import pandas as pd

import pretty_midi


def wav2mp3(filename_wav, remove_wav=True):
    filename_mp3 = filename_wav.replace('.wav', '.mp3')
    command = f'ffmpeg -i "{filename_wav}" -acodec libmp3lame "{filename_mp3}" -y -v 0'
    os.system(command)

    if os.path.exists(filename_mp3):
        if remove_wav:
            os.remove(filename_wav)
    else:
        print("Failed to convert from wav to mp3, check FFMPEG installment")


def synthesize_midi(path_dataset_processed):
    """Synthesize audio files based on midi files in the path

    Args:
        path_dataset_processed: path that has subfolders such as train/midi/

    """
    for split in ["train", "validation", "test"]:
        path_midi = os.path.join(path_dataset_processed, split, "midi")
        path_audio = os.path.join(path_dataset_processed, split, "midi_synthesized")

        os.makedirs(path_audio, exist_ok=True)

        piecenames = glob.glob(os.path.join(path_midi, "*.mid"))
        piecenames = [os.path.splitext(os.path.basename(x))[0] for x in piecenames]
        piecenames.sort()

        for piecename in piecenames:
            print(piecename)

            filename_midi = os.path.join(path_midi, piecename + ".mid")
            filename_audio = os.path.join(path_audio, piecename + ".wav")
            if os.path.exists(filename_audio.replace(".wav", ".mp3")):
                continue
            midi_data = pretty_midi.PrettyMIDI(filename_midi)
            wav = midi_data.fluidsynth()
            sf.write(filename_audio, wav, 44100, subtype='PCM_24')

            wav2mp3(filename_audio)


def load_midi_track_group_info_midiprogram(filename):
    """todo: @bochen please write some docstring here

    Args:
        filename: str, e.g., './jointist/dataset_creation/midi_track_group_config4.csv'

    Return:

    """

    midi_group_info = [x.strip() for x in open(filename, "r").readlines()]
    midi_group_info = midi_group_info[1:]  # remove the headline
    closed_set_inst_name = [
        x.split(",")[3].strip() for x in midi_group_info
    ]  # ["Piano", "Piano", "Piano", "Organ", "Organ", ...]
    program_number_mapping = [int(x.split(",")[4].strip()) for x in midi_group_info]

    print("closed set: ")
    print(set(closed_set_inst_name))

    return np.array(program_number_mapping)


def load_midi_track_group_info_plugin(filename):
    r"""Build plugin to program mapping.

    Args:
        filename: str, path of plugin config csv file, e.g.,
            './jointist/dataset_creation/midi_track_group_config4.csv'

    Returns:
        plugin_to_closed_set, dict, e.g.,
            {'AGML2': 0, 'bassoon': 0, ...}

    """

    df = pd.read_csv(filename, sep=',')

    plugin_names_num = len(df)
    plugin_to_closed_set = {}

    for k in range(plugin_names_num):
        plugin_name = df['Plugin_name'][k]
        program_number = df['Mapped MIDI program number'][k]
        plugin_to_closed_set[plugin_name] = program_number

    print("plugin_to_closed_set: ")
    print(plugin_to_closed_set)
    # E.g., {'AGML2': 0, 'bassoon': 0, ...}

    return plugin_to_closed_set


def group_midi_tracks_midiprogram(path_dataset, path_dataset_processed, program_number_mapping):
    """
    Group midi tracks based on MIDI program number.

    Args:
        path_dataset: The path of the original dataset
        path_dataset_processed: The path of the processed dataset
        program_number_mapping:

    Returns:

    """
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

            # get the trackname
            path_midi = os.path.join(path_dataset_split, piecename, "MIDI")
            tracknames = glob.glob(os.path.join(path_midi, "*.mid"))
            tracknames = [os.path.splitext(os.path.basename(x))[0] for x in tracknames]
            tracknames.sort()  # ["S01", "S02", "S03", ...]

            # placeholder for the processed MIDI
            filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", tracknames[0] + ".mid")
            midi_data_new = pretty_midi.PrettyMIDI(filename_midi)
            midi_data_new.instruments = []
            program_number_mapping = program_number_mapping[
                program_number_mapping > -1
            ]  # remove the -1 for placeholder
            new_midi_program_number_set = set(program_number_mapping)

            # program: program number for the processed MIDI: piano, organ, bass, strings, drums
            for program in new_midi_program_number_set:
                midi_data_new.instruments.append(pretty_midi.Instrument(program=program))
            # then add drum track, as always the last track
            midi_data_new.instruments.append(pretty_midi.Instrument(program=0))
            midi_data_new.instruments[-1].is_drum = True

            for trackname in tracknames:

                filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", trackname + ".mid")
                midi_data = pretty_midi.PrettyMIDI(filename_midi)
                if len(midi_data.instruments) > 1:
                    raise Exception("multi-track midi")
                instr = midi_data.instruments[0]

                if instr.is_drum:
                    for note in instr.notes:
                        # the last track is always drums
                        midi_data_new.instruments[-1].notes.append(note)
                else:
                    # [0, 16, 33, 48, ..., -1]
                    new_program_number = program_number_mapping[instr.program]
                    # -1 means "None", e.g., somd sound effects have no mapping
                    if new_program_number < 0:
                        continue
                    for instr_new in midi_data_new.instruments[:-1]:
                        # which track in placeholder should be grouped into (excluding drum track)
                        if instr_new.program == new_program_number:
                            for note in instr.notes:
                                instr_new.notes.append(note)

            for instr_new in midi_data_new.instruments[:-1]:  # manually lower an octave for bass
                new_program_number = instr_new.program
                if new_program_number in range(32, 40):
                    for note in instr_new.notes:
                        note.pitch -= 12

            filename_midi_out = os.path.join(path_midi_out, piecename + ".mid")
            midi_data_new.write(filename_midi_out)


def group_midi_tracks_plugin(
    path_dataset: str,
    path_dataset_processed: str,
    plugin_to_closed_set: str,
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
            filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", tracknames[0] + ".mid")
            closed_set_program_numbers = list(plugin_to_closed_set.values())
            closed_set_program_numbers = [int(x) for x in closed_set_program_numbers]
            closed_set_program_numbers = np.array(closed_set_program_numbers)
            closed_set_program_numbers = closed_set_program_numbers[
                closed_set_program_numbers > -1
            ]  # remove the -1 for placeholder

            closed_set_program_numbers = set(closed_set_program_numbers)
            # E.g., {0, 16, 30, 33, 48}

            midi_data_new = pretty_midi.PrettyMIDI(filename_midi)
            midi_data_new.instruments = []

            # Program number for the processed MIDI. E.g., {0, 16, 30, 33, 48} for piano, pizz, guitar, bass, drums.
            for program_number in closed_set_program_numbers:
                midi_data_new.instruments.append(pretty_midi.Instrument(program=program_number))

            # Then add drum track, as always the last track.
            midi_data_new.instruments.append(pretty_midi.Instrument(program=0))
            midi_data_new.instruments[-1].is_drum = True

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

                new_program_number = int(plugin_to_closed_set[plugin_name])

                if new_program_number < -1:  # is none
                    continue

                if new_program_number == -1:  # is drum
                    for note in instr.notes:
                        # the last track is always drums
                        midi_data_new.instruments[-1].notes.append(note)
                else:
                    # which track in placeholder should be grouped into (excluding drum track)
                    for instr_new in midi_data_new.instruments[:-1]:
                        
                        if instr_new.program == new_program_number:
                            for note in instr.notes:
                                if program_num in range(32, 40):
                                    note.pitch -= 12
                                instr_new.notes.append(note)

            '''
            # After combing all tracks, lower an octave for bass programs.
            for instr_new in midi_data_new.instruments[:-1]:
                new_program_number = instr_new.program
                if new_program_number in range(32, 40):
                    for note in instr_new.notes:
                        note.pitch -= 12
            '''
            filename_midi_out = os.path.join(path_midi_out, piecename + ".mid")
            midi_data_new.write(filename_midi_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_type', type=str, choices=["program", "plugin"], required=True)
    parser.add_argument('--config_csv_path', type=str, required=True)
    parser.add_argument('--path_dataset', type=str, required=True)
    parser.add_argument('--path_dataset_processed', type=str, required=True)

    args = parser.parse_args()
    config_type = args.config_type
    config_csv_path = args.config_csv_path
    path_dataset = args.path_dataset
    path_dataset_processed = args.path_dataset_processed

    if config_type == "program":
        program_number_mapping = load_midi_track_group_info_midiprogram(config_csv_path)
        group_midi_tracks_midiprogram(path_dataset, path_dataset_processed, program_number_mapping)

    if config_type == "plugin":
        plugin_to_closed_set = load_midi_track_group_info_plugin(config_csv_path)
        group_midi_tracks_plugin(path_dataset, path_dataset_processed, plugin_to_closed_set)

    else:
        raise NotImplementedError

    # run this if fluidsynth is installed, to synthesize the processed MIDIs for preview
    try:
        synthesize_midi(path_dataset_processed)
    except:
        print("Synthesize midi to audio fails! It does not matter!")
