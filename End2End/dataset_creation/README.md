
# Slakh dataset processing code

- To group the MIDI instruments to a closed set

## Files

- `midi_track_group_config_1.csv`: the config file of MIDI track group info. It groups to 5 tracks based on MIDI program number. The 5 tracks represent:
    - Piano
    - Organ
    - Strings
    - Bass
    - Drums
    
- `midi_track_group_config_2.csv`: the config file of MIDI track group info. It groups to 6 tracks based on plug-in name. The 6 tracks represent:
    - Piano
    - Organ
    - Strings
    - Bass
    - Distorted
    - Drums

- `midi_track_group_config_3.csv`: the config file of MIDI track group info. It groups to 3 tracks based on plug-in name. The 3 tracks represent:
    - Piano (all instruments other than bass and drum)
    - Bass
    - Drums

- `midi_track_group_config_4.csv`: the config file of MIDI track group info. It groups to 2 tracks based on plug-in name. The 2 tracks represent:
    - Piano (all pitched instruments)
    - Drums
    
- `prepare_closed_set.py`: the code to run dataset processing. Parameters are hardcoded in the main function entry, with comments.

## Environment

- `pip install pretty_midi`: to process MIDI files
- `pip install pyFluidSynth`: to synthesize the processed MIDI files for preview (optional)
- `ffmpeg`: to convert wav to mp3 (optional)

## The dataset

The original Slakh dataset is temporarily stored at:
https://www.dropbox.com/sh/5zh099o75kvpvz3/AADSqL8p3o7wLIvcta3IrP5Da?dl=0

The MIDI programs used in Slakh dataset are collected for listening at:
https://www.dropbox.com/sh/a0szdx51jl9p9i8/AADSvQnbQG_bmMfAMSsKR8UYa?dl=0

The sound plug-ins used in Slakh dataset are collected for listening at:
https://www.dropbox.com/sh/a2xw1t2h015nls0/AACJouypEmlgBPPK5EeHdqtxa?dl=0

