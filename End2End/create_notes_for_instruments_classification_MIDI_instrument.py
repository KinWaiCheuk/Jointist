import os
import numpy as np
import glob
import yaml
import soundfile as sf
import argparse
import pickle
import pathlib

import pretty_midi
from MIDI_program_map import idx2instrument_name


DRUMS_PLUGIN_NAMES = [
    'ar_modern_sparkle_kit_full',
    'ar_modern_white_kit_full',
    'funk_kit',
    'garage_kit_lite',
    'pop_kit',
    'session_kit_full',
    'stadium_kit_full',
    'street_knowledge_kit',
]


def create_notes(args):
    r"""Create list of notes information for instrument classification.

    Args:
        path_dataset: str, he path of the original dataset
        workspace: str
        split: str, 'train' | 'validation' | 'test'

    Returns:
        None
    """
    path_dataset = args.path_dataset
    workspace = args.workspace
    split = args.split

    # paths
    output_dir = os.path.join(workspace, 'instruments_classification_notes_MIDI_instrument', split)
    os.makedirs(output_dir, exist_ok=True)

    # MIDI file names.
    path_dataset_split = os.path.join(path_dataset, split)
    piecenames = os.listdir(path_dataset_split)
    piecenames = [x for x in piecenames if x[0] != "."]
    piecenames.sort()
    print("total piece number in %s set: %d" % (split, len(piecenames)))

    # output_list = []
    instrument_set = set()

    for n, piecename in enumerate(piecenames):
        print(n, piecename)

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

        note_event_list = []

        for trackname in tracknames:
            # E.g., "S00".

            plugin_name = metadata["stems"][trackname]["plugin_name"]
            program_num = metadata["stems"][trackname]["program_num"]
            

            plugin_name = os.path.splitext(os.path.basename(plugin_name))[0]
            # E.g., 'elektrik_guitar'.

            # Read MIDI file of a track
            filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", trackname + ".mid")
            midi_data = pretty_midi.PrettyMIDI(filename_midi)

            if len(midi_data.instruments) > 1:
                raise Exception("multi-track midi")

            instr = midi_data.instruments[0]

            # Append all notes of a track to output_list if not drums.
            if plugin_name not in DRUMS_PLUGIN_NAMES:
                instrument_set.add(program_num)

                for note in instr.notes:

                    # Lower an octave for bass.
                    if instr.program in range(32, 40):
                        pitch = note.pitch - 12
                    else:
                        pitch = note.pitch

                    note_event = {
                        'split': split,
                        'audio_name': piecename,
                        'plugin_name': idx2instrument_name[program_num],
                        'plugin_names': [idx2instrument_name[program_num]],
                        'start': note.start,
                        'end': note.end,
                        'pitch': pitch,
                        'velocity': note.velocity,
                    }

                    # Remove notes with MIDI pitches larger than 109 (very few).
                    if note.pitch < 109:
                        # output_list.append(note_event)
                        note_event_list.append(note_event)

        
        note_event_list.sort(key=lambda note_event: note_event['start'])

        note_event_list = add2(note_event_list)
        # output_list += note_event_list

        # E.g., output_list looks like: [
        #     {'split': 'train', 'audio_name': 'Track00001', 'plugin_name':
        #      'elektrik_guitar', 'start': 0.7811, 'end': 1.2576, 'pitch': 64, 'velocity': 127,
        #     },
        #     ...
        #     {'split': 'train', 'audio_name': 'Track00003', 'plugin_name':
        #      'jazz_guitar2', 'start': 58.2242, 'end': 58.4500, 'pitch': 57, 'velocity': 100,
        #     },
        #     ...
        # ]

        output_path = os.path.join(output_dir, '{}.pkl'.format(pathlib.Path(piecename).stem))
        pickle.dump(note_event_list, open(output_path, 'wb'))
        print('Write out to {}'.format(output_path))
    pickle.dump(instrument_set, open(f'{split}_set', 'wb'))



def add2(note_event_list):

    new_list = []

    for note_event in note_event_list:
        note_event['instruments_num'] = 1

    for i in range(1, len(note_event_list)):
        if note_event_list[i]['pitch'] == note_event_list[i - 1]['pitch']:
            if note_event_list[i]['start'] - note_event_list[i - 1]['start'] <= 0.05:

                new_plugin_names = note_event_list[i]['plugin_names'] + note_event_list[i - 1]['plugin_names']
                new_instruments_num = note_event_list[i - 1]['instruments_num'] + 1

                for j in range(note_event_list[i - 1]['instruments_num'] + 1):
                    note_event_list[i - j]['instruments_num'] = new_instruments_num
                    note_event_list[i - j]['plugin_names'] = new_plugin_names

    for note_event in note_event_list:
        if len(note_event['plugin_names']) > 1:
            plugin_names = list(set(note_event['plugin_names']))
            note_event['plugin_names'] = plugin_names
            note_event['instruments_num'] = len(plugin_names)

    # for i in range(1, len(note_event_list)):
    #     if len(note_event_list[i]['plugin_names']) == 3:
    #         from IPython import embed; embed(using=False); os._exit(0)

    # for i in range(5):
    #     cnt = 0
    #     for note_event in note_event_list:
    #         if note_event['instruments_num'] == i:
    #             cnt += 1
    #     print(cnt)
    
    return note_event_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('create_notes')
    parser_a.add_argument('--path_dataset', type=str, required=True)
    parser_a.add_argument('--workspace', type=str, required=True)
    parser_a.add_argument('--split', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'create_notes':
        create_notes(args)

    else:
        raise NotImplementedError

