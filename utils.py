import mirdata
import crepe
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def compute_dataset_stats(dataset):
    """
    Compute statistics for a given dataset, including the number of tracks, 
    distribution by instrument, and distribution by genre.

    Parameters
    ----------
    dataset : mirdata.core.Dataset
        The dataset object for which to compute statistics.

    Returns
    -------
    dict
        A dictionary containing:
        - 'num_tracks' (int): Total number of tracks in the dataset.
        - 'instrument_distribution' (dict): A dictionary where keys are instrument names
          and values are counts of tracks for each instrument.
        - 'genre_distribution' (dict): A dictionary where keys are genre names and values 
          are counts of tracks for each genre.

    Notes
    -----
    The function expects that the tracks in the dataset might have attributes named 
    'instrument' and 'genre'. If a track lacks these attributes, it's simply not 
    considered in the respective distribution.

    Example
    -------
    >>> dataset = mirdata.initialize("medleydb_pitch")
    >>> stats = compute_dataset_stats(dataset)
    >>> print(stats)
    {
        'num_tracks': 100,
        'instrument_distribution': {'flute': 30, 'guitar': 50, ...},
        'genre_distribution': {'jazz': 40, 'rock': 30, ...},
    }
    """
    # YOUR CODE HERE

    # get total number of tracks
    stats_dict = {}
    stats_dict['num_tracks'] = len(dataset.track_ids)

    # use Counter() to make things easier
    inst_cnt = Counter()
    genre_cnt = Counter()

    # iterate over tracks for instrument/genre stats

    # this method didn't pass the pytests due to the way they were set up,
    #  so i had to make things a bit uglier

    # for track_id, track in tracks.items():
    #     inst_cnt[track.instrument] += 1
    #     genre_cnt[track.genre] += 1

    for track_id in dataset.track_ids:
        try: 
            instrument = dataset.track(track_id).instrument
            genre = dataset.track(track_id).genre
            inst_cnt[instrument] += 1
            genre_cnt[genre] += 1
        except:
            pass

    stats_dict['instrument_distribution'] = dict(inst_cnt)
    stats_dict['genre_distribution'] = dict(genre_cnt)
        
    return stats_dict



def estimate_pitch(audio, sr, voicing_threshold=0.3, use_viterbi=False):
    """
    Estimate the fundamental frequency (pitch) of an audio file using the CREPE algorithm.

    Parameters
    ----------
    audio_path : str
        The file path to the input audio file.
    voicing_threshold : float, optional
        The confidence threshold above which a frame is considered voiced. Frames with confidence
        levels below this threshold are marked as unvoiced (i.e., set to 0 Hz).
        Default is 0.3.
    use_viterbi : bool, optional
        If True, apply Viterbi decoding to smooth the pitch track and obtain more consistent
        pitch estimates over time. Default is False.

    Returns
    -------
    time : np.ndarray
        A 1D numpy array containing time stamps for each frame in seconds.
    frequency : np.ndarray
        A 1D numpy array containing the estimated pitch for each frame in Hz. Unvoiced frames
        are set to 0 Hz.
    confidence : np.ndarray
        A 1D numpy array containing the confidence of the pitch estimate for each frame.
    activation : np.ndarray
        A 2D numpy array representing the activation matrix returned by the CREPE algorithm,
        which can be used to visualize the pitch estimation process.

    """

    # Hint: follow this steps
    # load audio using librosa
    # use crepe.predict 
    # you will need to do a little postprocessing before returning the 
    # frequency values, remember that you need to determine the voicing first looking at the activation, 
    # and then which is the most likely frequency for the voiced frames.
    # read 
    
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=use_viterbi)
    for i, confidence_level in enumerate(confidence):
        if confidence_level < voicing_threshold:
            frequency[i] = 0

    # YOUR CODE HERE
    return time, frequency, confidence, activation

def evaluate_pitch(data, voicing_threshold=0.3, use_viterbi=False):
    """
    Evaluate pitch estimation for multiple tracks using mir_eval.
    
    Parameters
    ----------
    data : dict
        Dictionary containing track information. Keyed by track ID with values being track objects.
        Each track object is expected to have an `audio_path` attribute for the audio file and a 
        `pitch` attribute which has `times` and `frequencies` attributes.
    voicing_threshold : float, optional
        Threshold on the voicing to determine which frames are unvoiced. Defaults to 0.3.
    use_viterbi : bool, optional
        If True, use the Viterbi algorithm during pitch estimation. Defaults to False.

    Returns
    -------
    dict
        Dictionary containing evaluation scores for each track. Keyed by track ID with values being
        the evaluation results from mir_eval.melody.evaluate (which is a dictionary).

    Notes
    -----
    This function makes use of the `estimate_pitch` function to estimate the pitch for each track,
    and then evaluates the estimated pitch against the ground truth using mir_eval.
    """
    eval_scores = {}
    for track_id, track in data.items():
        est_time, est_freq, est_confidence, est_activation = estimate_pitch(
                track.audio_path, voicing_threshold=voicing_threshold, use_viterbi=use_viterbi
                )

        ref_time = track.pitch.times
        ref_freq = track.pitch.frequencies
        score = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
        eval_scores[track_id] = score

    return eval_scores


def prepare_boxplot_data(pitch_scores):
    """
    Prepare pitch tracking evaluation scores for boxplot visualization.

    Parameters
    ----------
    pitch_scores : dict
        A dictionary where the keys are track names and the values are ordered dictionaries
        of evaluation metrics, such as Voicing Recall, Voicing False Alarm, Raw Pitch Accuracy,
        Raw Chroma Accuracy, and Overall Accuracy.

    Returns
    -------
    data_dict : dict
        A dictionary where each key is an evaluation metric and the value is a list of all
        scores for that metric across all tracks. Suitable for use in creating boxplots.

    Examples
    --------
    >>> pitch_scores = {
    ...     'Track_1': OrderedDict([
    ...         ('Voicing Recall', 0.99),
    ...         ('Voicing False Alarm', 0.45),
    ...         ...
    ...     ]),
    ...     'Track_2': OrderedDict([
    ...         ('Voicing Recall', 0.98),
    ...         ('Voicing False Alarm', 0.50),
    ...         ...
    ...     ]),
    ...     ...
    ... }
    >>> data_dict = prepare_boxplot_data(pitch_scores)
    >>> for metric, scores in data_dict.items():
    ...     print(f"{metric}: {scores}")
    Voicing Recall: [0.99, 0.98, ...]
    Voicing False Alarm: [0.45, 0.50, ...]
    ...
    """
    # YOUR CODE HERE
    data_dict = defaultdict(list)

    for track, metrics in pitch_scores.items():
        for metric, score in metrics.items():
            data_dict[metric].append(score)

    return dict(data_dict)


def split_by_instrument(scores_dictionary, tracks_dictionary):
    """
    Split scores by instrument, retaining only the scores for the top 6 most frequently 
    occurring instruments.

    The function takes as input a dictionary of scores that have an associated instrument
    and returns a nested dictionary where the outer keys are instrument names and the 
    inner dictionaries are scores associated with unique track identifiers.

    Parameters
    ----------
    scores_dictionary : dict
        A dictionary where keys are track IDs and values are objects or dictionaries 
        that have an 'instrument' attribute or key and a 'score' attribute or key. 
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    instrument_scores : dict
        A dictionary with instrument names as keys. Each key maps to another dictionary
        containing track IDs as keys and their associated scores as values.

    Example
    --------

    Dictionary should look like:
    {'male singer': {'AClassicEducation_NightOwl_STEM_08': OrderedDict([('Voicing Recall',
                0.9981117230527145),
               ('Voicing False Alarm', 0.46255349500713266),
               ('Raw Pitch Accuracy', 0.9851298190401259),
               ('Raw Chroma Accuracy', 0.9853658536585366),
               ('Overall Accuracy', 0.7301076725130359)]),
                'AClassicEducation_NightOwl_STEM_13': OrderedDict([('Voicing Recall',
                0.995873786407767),
               ('Voicing False Alarm', 0.8500986193293886), ....}
    """
    instrument_count = Counter()

    for track_id, score_data in scores_dictionary.items():
        instrument = tracks_dictionary[track_id].instrument
        instrument_count[instrument] += 1

    top_6_instruments = [instrument for instrument, _ in instrument_count.most_common(6)]

    instrument_scores = defaultdict(dict)

    for track_id, score_data in scores_dictionary.items():
        instrument = tracks_dictionary[track_id].instrument
        if instrument in top_6_instruments:
            instrument_scores[instrument][track_id] = score_data

    return dict(instrument_scores)



def split_by_genre(scores_dictionary, tracks_dictionary):
    """Split scores by genre.

    Parameters
    ----------
    scores_dictionary : dict
        Dictionary of scores keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    genre_scores : dict
        Dictionary with genre as keys and a
        dictionary of scores keyed by track_id as values

    Example
    -------

    Dictionary should look like:
    {'Singer/Songwriter': {'AClassicEducation_NightOwl_STEM_08': OrderedDict([('Voicing Recall',
                0.9981117230527145),
               ('Voicing False Alarm', 0.46255349500713266),
               ('Raw Pitch Accuracy', 0.9851298190401259),
               ('Raw Chroma Accuracy', 0.9853658536585366),
               ('Overall Accuracy', 0.7301076725130359)]),
                'AClassicEducation_NightOwl_STEM_13': OrderedDict([('Voicing Recall',
                0.995873786407767),
               ('Voicing False Alarm', 0.8500986193293886),
               ('Raw Pitch Accuracy', 0.9854368932038835), ....}

    """

    genre_scores = defaultdict(dict)

    for track_id, score_data in scores_dictionary.items():
        instrument = tracks_dictionary[track_id].genre
        genre_scores[instrument][track_id] = score_data

    return dict(genre_scores)


def plot_pitch(time, frequency, confidence, activation):
    """
    Plot pitch tracking information including the fundamental frequency (F0) over time, 
    the confidence of the estimates, and an activation matrix representing the salience 
    of pitches over time.

    Parameters
    ----------
    time : array_like
        An array of time stamps at which the frequency and confidence values are estimated.
    frequency : array_like
        An array containing estimated fundamental frequency (F0) values in Hertz (Hz) for each time stamp.
    confidence : array_like
        An array containing confidence values associated with each F0 estimate.
    activation : array_like
        A 2D array representing the activation of different pitch bins over time. 
        The vertical dimension corresponds to pitch bins, and the horizontal dimension 
        corresponds to time.

    Notes
    -----
    This function plots three subplots: The first subplot displays the F0 estimate over time,
    the second subplot shows the confidence of these estimates over time, and the third 
    subplot shows the activation matrix with pitch bins in cents over time. A bug fix is 
    applied for the pitch calculation as per a known issue in the CREPE repository.

    The function does not return any values but renders a matplotlib figure directly.

    References
    ----------
    .. [1] https://github.com/marl/crepe/issues/2
    """
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(12, 8), sharex=False)
    axes[0].plot(time, frequency)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Estimated F0 (Hz)")
    axes[0].set_title("F0 Estimate Over Time")
    
    axes[1].plot(time, confidence)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Estimate Confidence Over Time")
    
    axes[2].imshow(activation.T, origin="lower", aspect="auto")
    axes[2].set_xticks(np.arange(len(activation))[::500])
    
    c1 = 32.7 # Hz, fix for a known issue in CREPE
    c1_cent = mir_eval.melody.hz2cents(np.array([c1]))[0]
    c = np.arange(0, 360) * 20 + c1_cent
    freq = 10 * 2 ** (c / 1200)
    
    axes[2].set_yticks(np.arange(len(freq))[::35])
    axes[2].set_yticklabels([int(f) for f in freq[::35]])
    axes[2].set_ylim([0, 300])
    axes[2].set_xticklabels((np.arange(len(activation))[::500] / 100).astype(int))
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Activation Matrix: 20 Cent Bins Over Time")
    
    plt.tight_layout()
    plt.show()
