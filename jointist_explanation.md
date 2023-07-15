# Inference code explanation
The main inference code is located at  the `predict_step()` function under the `Transcription` class in `End2End/tasks/transcription/transcription.py`. It involves two steps:

`waveforms => piano rolls => pkl/midi files`


## Step 1: Converting audio clips into posteriorgrams (or piano rolls)
During inference, audio clips can be of any lengths. We specify the `segment_samples` to determine how long each segment is, and use `segment_samples//2` as the hop size to cut long audio clips into something that our model can handle. For example, a 4-minute audio clip would be a waveform of the shape (4*60*16000)=(3840000). And `segment_samples=160,000` will generate waveforms segments of the shape (47, 160000). Then we use `seg_batch_size` to control how many segments are fed to the network per batch. If `seg_batch_size=8`, `47/8=6` feed-forwards are needed to finish transcribing the audio.

`predict_probabilities()` at line 429 of `End2End/tasks/transcription/transcription.py` is responsible for this operation:

```python
predict_probabilities(
    network,
    audio,
    condition,
    segment_samples
    seg_batch_size
    )
```

**network**: The pytorch model for feedforward

**audio**: Single waveform (batch size must be 1 during inference) of the shape (len). 

**condition**: One-hot vector corresponding to the instruments of the shape (39)

**segment_samples**: The length of each audio segment (default is 10 seconds/160,000 samples)

**seg_batch_size**: How many segments per each feedforward (default is 8)

This function returns a dictionary called `_output_dict` containing  two keys

```python
_output_dict={
    'frame_output': (num_frames, 88),
    'reg_onset_output': (num_frames, 88),
    }
```
 
Each `_output_dict` from `predict_probabilities()` at line 429 of `End2End/tasks/transcription/transcription.py` corresponds to one musical instrument.

After that, we concatenate all the outputs for different instrument forming a new dictionary `output_dict` at line 435 of `End2End/tasks/transcription/transcription.py`.

```python
output_dict={
    'frame_output': (num_frames, 88*num_instruments),
    'reg_onset_output': (num_frames, 88*num_instruments),    
}
```
 
 
`frame_output` and `reg_onset_output` are posteriorgrams with values between [0,1] indiciting the probability of the notes are present. Piano rolls can be easily obtained by applying a thershold to `frame_output`. In jointist, we use a onset and frame postprocessor `OnsetFramePostProcessor()` to directly convert these posteriorgrams into pkl/midi files via `postprocess_probabilities_to_midi_events()`.


## Step 2: Converting posteriorgrams into pkl/midi files
After we have all the posteriorgrams for all instruments stored in `output_dict`, we use `postprocess_probabilities_to_midi_events()` to obtain the pkl/midi files

```
midi_events=postprocess_probabilities_to_midi_events(
    output_dict,
    plugin_ids,
    IX_TO_NAME,
    classes_num,
    post_processor)
```

**output_dict**: The output obtained from line 435 of `End2End/tasks/transcription/transcription.py`.

**plugin_ids**: a list of indices indicting what instruments to be transcribed

**IX_TO_NAME**: The dictionary mapping indices back to its instrument names in string

**classes_num**: number of instrument classes (39) 

**post_processor**: Different post-processor avaliable in `End2End/inference_instruments_filter.py`. Default is `OnsetFramePostProcessor()`