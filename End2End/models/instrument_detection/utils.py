import torch
import numpy as np

def summarized_output(segmentwise_output, threshold=0.5):
    """
    input:
        segmentwise_output: (N, output_classes)
    output:
        (output_classes)
    """
    
    bool_pred = torch.sigmoid(segmentwise_output)>threshold
    bool_summarized = torch.zeros(bool_pred.shape[1]).to(bool_pred.device)
    
    for i in bool_pred:
        bool_summarized = torch.logical_or(bool_summarized,  i)    
        
    return bool_summarized


def obtain_segments(audio, segment_samples):
    # Preparing placeholders for audio segmenting
    audio_length = audio.shape[1]
    # Pad audio to be evenly divided by segment_samples.
    pad_len = int(np.ceil(audio_length / segment_samples)) * segment_samples - audio_length
    
    if audio_length>segment_samples:
        audio = torch.cat((audio, torch.zeros((1, pad_len), device=audio.device)), axis=1)

        # Enframe to segments.
        segments = audio.unfold(1, segment_samples, segment_samples//2).squeeze(0) # faster version of enframe
        # (N, segment_samples)
        return segments, audio_length
    else:
        return audio, audio_length