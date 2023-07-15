import numpy as np


def note_detection_with_onset_offset_regress(
    frame_output,
    onset_output,
    onset_shift_output,
    offset_output,
    offset_shift_output,
    velocity_output,
    frame_threshold,
):
    r"""Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            # Onset detected.
            if bgn:
                # Consecutive onsets. E.g., pedal is not released, but two
                # consecutive notes being played.
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                # output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, 0.8])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            # If onset found, then search offset.
            if frame_output[i] <= frame_threshold and not frame_disappear:
                # Frame disappear detected.
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                # Offset detected.
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    # bgn --------- offset_occur --- frame_disappear.
                    fin = offset_occur
                else:
                    # bgn --- offset_occur --------- frame_disappear.
                    fin = frame_disappear
                output_tuples.append(
                    [
                        bgn,
                        fin,
                        onset_shift_output[bgn],
                        offset_shift_output[fin],
                        velocity_output[bgn],
                    ]
                )
                # output_tuples.append(
                # [bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], 0.8]
                # )
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                # Offset not detected.
                fin = i
                output_tuples.append(
                    [
                        bgn,
                        fin,
                        onset_shift_output[bgn],
                        offset_shift_output[fin],
                        velocity_output[bgn],
                    ]
                )
                # output_tuples.append(
                # [bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], 0.8]
                # )
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def note_detection_with_onset_regress(
    frame_output,
    onset_output,
    onset_shift_output,
    velocity_output,
    frame_threshold,
):
    r"""Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    for bgn in np.argwhere(onset_output):
        condition=True 
        bgn = bgn[0] # convert numpy array into int
        i = bgn + 1 
        if i == len(onset_output):
            pass # if onset appears at the last timestep, skip
        else:
            while frame_output[i] > frame_threshold and condition:

                if onset_output[i] == 1:           
                    # Onset detected.
                    # Consecutive onsets. E.g., pedal is not released, but two
                    # consecutive notes being played.
                    fin = max(i, 0) # i-1 doesn't work when the next timestep is another onset
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                    condition = False

                elif (i - bgn >= 600 or i == onset_output.shape[0] - 1):                             
                    # Offset not detected.
                    fin = i # This is to prevent floating point error, ensuring the exact same result as the original code
                    output_tuples.append(
                        [
                            bgn,
                            fin,
                            onset_shift_output[bgn],
                            0,
                            velocity_output[bgn],
                        ]
                    )
                    condition = False

                else:
                    i += 1
                    if onset_output[i] == 1:
                        i-=1 # If note disappear and onset appear at the same time, do this to ensure the result is same as the old version
                        break       

            
        if condition:              
            fin=i
            output_tuples.append(
                [
                    bgn,
                    fin,
                    onset_shift_output[bgn],
                    0,
                    velocity_output[bgn],
                ]
            )


    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples

def note_detection_with_onset_regress2(
    frame_output,
    onset_output,
    low_onset_output,
    onset_shift_output,
    velocity_output,
    frame_threshold,
):
    r"""Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    refactory_time = 0

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            # Onset detected.
            if bgn:
                # Consecutive onsets. E.g., pedal is not released, but two
                # consecutive notes being played.
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                # output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, 0.8])
                frame_disappear = None

            bgn = i

        if bgn:
            if i > bgn:
                # If onset found, then search offset.
                if frame_output[i] <= frame_threshold and not frame_disappear:
                    # Frame disappear detected.
                    frame_disappear = i

                if frame_disappear:
                    fin = frame_disappear
                    output_tuples.append(
                        [
                            bgn,
                            fin,
                            onset_shift_output[bgn],
                            0,
                            velocity_output[bgn],
                        ]
                    )
                    # output_tuples.append(
                    # [bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], 0.8]
                    # )
                    bgn, frame_disappear = None, None
                    refactory_time = fin + 100

                if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                    # Offset not detected.
                    fin = i
                    output_tuples.append(
                        [
                            bgn,
                            fin,
                            onset_shift_output[bgn],
                            0,
                            velocity_output[bgn],
                        ]
                    )
                    # output_tuples.append(
                    # [bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], 0.8]
                    # )
                    bgn, frame_disappear = None, None
                    refactory_time = fin + 100

        else:
            # from IPython import embed; embed(using=False); os._exit(0)
            if i > refactory_time and np.max(frame_output[max(i - 5, 0) : min(i + 5, onset_output.shape[0])]) >= 0.5:
                for j in range(50):
                    if i - j >= 0:
                        if low_onset_output[i - j] == 1:
                            bgn = i - j
                            frame_disappear = None
                            # if i < 500:
                            #     from IPython import embed; embed(using=False); os._exit(0)
                            break

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def pedal_detection_with_onset_offset_regress(frame_output, offset_output, offset_shift_output, frame_threshold):
    r"""Process prediction array to pedal events information.

    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift],
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533],
        [1909, 1947, 0.30730522, -0.45764327],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            # Pedal onset detected.
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            # If onset found, then search offset.
            if frame_output[i] <= frame_threshold and not frame_disappear:
                # Frame disappear detected.
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                # Offset detected.
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                # offset not detected but frame disappear.
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def drums_detection_with_onset_regress(
    onset_output,
    onset_shift_output,
    velocity_output,
):
    r"""Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            # Onset detected.
            if bgn:
                # Consecutive onsets. E.g., pedal is not released, but two
                # consecutive notes being played.
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                # output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, 0.8])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:

            if bgn and (i - bgn >= 10 or i == onset_output.shape[0] - 1):
                # Offset not detected.
                fin = i
                output_tuples.append(
                    [
                        bgn,
                        fin,
                        onset_shift_output[bgn],
                        0,
                        velocity_output[bgn],
                    ]
                )
                # output_tuples.append(
                # [bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], 0.8]
                # )
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples
