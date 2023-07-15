import pandas as pd
import json
import pathlib


SAMPLE_RATE = 16000
CLASSES_NUM = 88  # Number of notes of piano
BEGIN_NOTE = 21  # MIDI note of A0, the lowest note of a piano.
SEGMENT_SECONDS = 10.0  # Training segment duration
HOP_SECONDS = 1.0
FRAMES_PER_SECOND = 100
VELOCITY_SCALE = 128

TAGGING_SEGMENT_SECONDS = 2.0

# Load plugin related information.
with open('End2End/dataset_creation/plugin_to_midi_program.json') as f:
    plugin_dict = json.load(f)

PLUGIN_LABELS = sorted([pathlib.Path(key).stem for key in plugin_dict.keys()])
# E.g., ['AGML2', ..., 'bass_trombone', 'bassoon', ...]

PLUGIN_LABELS_NUM = len(PLUGIN_LABELS)
PLUGIN_LB_TO_IX = {lb: i for i, lb in enumerate(PLUGIN_LABELS)}
PLUGIN_IX_TO_LB = {i: lb for i, lb in enumerate(PLUGIN_LABELS)}

# Get plugin name to instruments mapping.
PLUGIN_NAME_TO_INSTRUMENT = {}

for key in plugin_dict.keys():
    count = -1

    for instrument_name in plugin_dict[key].keys():
        this_count = plugin_dict[key][instrument_name]

        if this_count > count:
            instrument = instrument_name
            count = this_count

    PLUGIN_NAME_TO_INSTRUMENT[pathlib.Path(key).stem] = instrument

# E.g., PLUGIN_NAME_TO_INSTRUMENT: {
#    'elektrik_guitar': 'Overdriven Guitar',
#    'session_kit_full': 'Drums',
#    ...
# }

BN_MOMENTUM = 0.01  # a globally applied momentum
