WORKSPACE="/opt/tiger/kinwai/jointist"
DATASET_DIR="${WORKSPACE}/openmic-2018"
WAVEFORM_HDF5S_DIR="${WORKSPACE}/hdf5s/openmic_waveforms"

# # Download audio files
# wget https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz?download=1 ./
# tar -xvf openmic-2018-v1.0.0.tgz\?download\=1

# # pack audios
# # Pack audio files into hdf5 files.
python3 End2End/create_openmic2018.py pack_audios_to_hdf5s \
    --audios_dir=$DATASET_DIR \
    --hdf5s_dir=$WAVEFORM_HDF5S_DIR

# The labels are inside the csv file, no need to create pkl files for it