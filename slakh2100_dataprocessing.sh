WORKSPACE=$PWD
SLAKH_DATASET_DIR="${WORKSPACE}/datasets/slakh2100/slakh2100_flac"
WAVEFORM_HDF5S_DIR="${WORKSPACE}/hdf5s/waveforms"

# Download audio files
./scripts/dataset-slakh2100/download_slakh2100_from_hdfs.sh

# pack audios
# Pack audio files into hdf5 files.
python3 End2End/dataset_creation/create_slakh2100.py pack_audios_to_hdf5s \
    --audios_dir=$SLAKH_DATASET_DIR \
    --hdf5s_dir=$WAVEFORM_HDF5S_DIR
    
#Create pkl files for instrument classificatin
for SPLIT in 'train' 'validation' 'test'
do
    python3 End2End/create_notes_for_instruments_classification_MIDI_class.py create_notes \
        --path_dataset=$SLAKH_DATASET_DIR \
        --workspace=$WORKSPACE \
        --split=$SPLIT
done

# # ====== Train piano roll transcription
# # Prepare slakh2100 into piano + drums MIDI files.
# AUDIOS_DIR="/opt/tiger/debugtest/jointist/datasets/slakh2100/slakh2100_flac"
# CONFIG_NAME="config_4"  # Piano + drums
# CONFIG_CSV_PATH="./jointist/dataset_creation/midi_track_group_${CONFIG_NAME}.csv"
# PATH_DATASET_PROCESSED="${WORKSPACE}/dataset_processed/closed_set_${CONFIG_NAME}"
# python3 jointist/dataset_creation/prepare_closed_set.py \
#     --config_type="plugin" \
#     --config_csv_path=$CONFIG_CSV_PATH \
#     --path_dataset=$AUDIOS_DIR \
#     --path_dataset_processed=$PATH_DATASET_PROCESSED

# # Pack MIDI events
# PROCESSED_MIDIS_DIR="${WORKSPACE}/dataset_processed/closed_set_config_4"
# MIDI_EVENTS_HDF5S_DIR="${WORKSPACE}/pickles/prettymidi_events/closed_set_config_4"
# python3 jointist/dataset_creation/create_slakh2100.py pack_midi_events_to_hdf5s \
#     --processed_midis_dir=$PROCESSED_MIDIS_DIR \
#     --hdf5s_dir=$MIDI_EVENTS_HDF5S_DIR