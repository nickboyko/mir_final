import librosa
import mir_eval
import utils as u
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mir_eval import display as mir_display


# ================== AUXILIARY FUNCTIONS ===========================

# DO NOT MODIFY THIS FUNCTIONS, THEY ARE FOR AUXILIARY USE


def plot_distributions(stats):
    """
    Display bar charts for the distribution of instruments and genres.

    This function takes a dictionary containing instrument and genre distribution
    data and plots two bar charts: one for the instrument distribution and another
    for the genre distribution. Each bar chart shows the counts of items in each
    category (instruments or genres).

    Parameters
    ----------
    stats : dict
        A dictionary with two keys: 'instrument_distribution' and 'genre_distribution'.
        Each key should have a dictionary as its value, mapping the category names to their counts.

        Example:
        {
            'instrument_distribution': {'Piano': 10, 'Violin': 5, ...},
            'genre_distribution': {'Jazz': 7, 'Rock': 3, ...}
        }

    Returns
    -------
    None
        This function does not return any value. It displays the plots directly.

    """
 
    # Plot instrument distribution
    instrument_names = list(stats['instrument_distribution'].keys())
    instrument_counts = list(stats['instrument_distribution'].values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(instrument_names, instrument_counts, color='blue')
    plt.title('Instrument Distribution')
    plt.xlabel('Instruments')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot genre distribution
    genre_names = list(stats['genre_distribution'].keys())
    genre_counts = list(stats['genre_distribution'].values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(genre_names, genre_counts, color='green')
    plt.title('Genre Distribution')
    plt.xlabel('Genres')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


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


def plot_comparison_boxplot(data_no_viterbi, data_viterbi):
    """
    Plot separate side-by-side boxplots for each pitch tracking metric to compare 'No Viterbi' vs 'Viterbi' data.

    Parameters
    ----------
    data_no_viterbi : dict
        A dictionary with keys as metric names and values as lists of scores without Viterbi decoding.
    data_viterbi : dict
        A dictionary with keys as metric names and values as lists of scores with Viterbi decoding.

    Returns
    -------
    None
    """

    # Number of metrics to plot
    num_metrics = len(data_no_viterbi)

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(1, num_metrics, figsize=(16, 6))

    # Go through each metric and create its own subplot
    for i, (metric, scores_no_viterbi) in enumerate(data_no_viterbi.items()):
        scores_viterbi = data_viterbi[metric]

        # Define positions for 'No Viterbi' and 'Viterbi'
        positions_no_viterbi = [1]
        positions_viterbi = [2]

        # Create boxplots for 'No Viterbi' and 'Viterbi'
        bp_no_viterbi = axs[i].boxplot(scores_no_viterbi, positions=positions_no_viterbi,
                                        patch_artist=True, boxprops=dict(facecolor='cyan'))

        bp_viterbi = axs[i].boxplot(scores_viterbi, positions=positions_viterbi,
                                     patch_artist=True, boxprops=dict(facecolor='lightgreen'))

        # Set the titles and labels
        axs[i].set_title(metric)
        axs[i].set_xticks([1.5])
        axs[i].set_xticklabels(['Comparison'])

        # If it's the first subplot, add the y-label and legend
        if i == 0:
            axs[i].set_ylabel('Scores')
            axs[i].legend([bp_no_viterbi["boxes"][0], bp_viterbi["boxes"][0]], ['No Viterbi', 'Viterbi'], loc='upper right')

    # Adjust the layout
    plt.tight_layout()
    plt.show()

def plot_per_category(category_scores, metric):
    """
    Plot a boxplot for a specified metric across different categories.

    This function takes a dictionary with categories as keys and lists of scores as values.
    It generates a boxplot for a given metric, with one box for each category. The categories
    are displayed along the x-axis, and the metric scores are displayed along the y-axis.

    Parameters
    ----------
    category_scores : dict
        A dictionary where each key is a category (e.g., a genre, instrument) and each value
        is a list of dictionaries containing different metric scores for items in that category.
        Each inner dictionary should have a key that matches the specified metric.
    
    metric : str
        The name of the metric to plot. This string must match one of the keys in the
        dictionaries that are the values of the `category_scores` dictionary.

    Returns
    -------
    None
        This function does not return any values. It displays the boxplot directly.

    """

    # Flatten the category scores into a list of scores for each category
    data_to_plot = []
    category_names = []
    
    for category, data in category_scores.items():
        data_boxplot = u.prepare_boxplot_data(data)
        scores = data_boxplot[metric]
        data_to_plot.append(scores)
        category_names.append(category)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a colormap
    colors = cm.viridis(np.linspace(0, 1, len(category_scores)))

    # Create the boxplot
    boxplots = ax.boxplot(data_to_plot, patch_artist=True)

    # Set colors for each boxplot
    for patch, color in zip(boxplots['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set the x-tick labels to the names of the categorys
    ax.set_xticks(range(1, len(category_names) + 1))
    ax.set_xticklabels(category_names, rotation=45, ha='right')

    # Set the titles and labels
    ax.set_title(f'Boxplot of {metric}')
    ax.set_ylabel('Scores')
    
    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

import numpy as np
from IPython.display import Audio, display
import mir_eval


def plot_and_sonify_outputs(track_id, data, viterbi=False):
    """
    Plot reference and estimated pitches for a track and directly play the sonified pitch contours
    mixed with the original audio.

    Parameters:
    - track_id (str): The identifier for the track in the dataset.
    - data (dict): The dataset containing pitch and audio path information.
    - viterbi (bool): Whether to use Viterbi decoding in pitch estimation.

    This function creates a plot and plays audio for the specified track.
    The plot displays the reference pitch (in black) against the estimated pitch (in red).
    The audio streams contain the original audio mixed with the sonified estimated pitch
    and the sonified reference pitch, respectively.
    """

    # Estimate the pitch using the provided data and settings
    time, freq, conf, act = u.estimate_pitch(data[track_id].audio_path, use_viterbi=viterbi)
    time_ref = data[track_id].pitch.times
    freq_ref = data[track_id].pitch.frequencies
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 7))
    mir_display.pitch(time_ref, freq_ref, color="k", linewidth=2.1, label="Reference", ax=ax)
    mir_display.pitch(time, freq, color="r", linewidth=0.8, label="Estimate", ax=ax)
    ax.legend()
    ax.set_title(f"Pitch Comparison for {track_id}" + (" with Viterbi" if viterbi else ""))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.show()

    # Sonify the estimated pitch contour
    son_est = mir_eval.sonify.pitch_contour(time, freq, fs=44100)
    
    # Sonify the reference pitch contour
    son_ref = mir_eval.sonify.pitch_contour(time_ref, freq_ref, fs=44100)
    
    # Load the original audio
    original_audio, sr = librosa.load(data[track_id].audio_path, sr=None)

    # Trim or pad the original audio to match the sonification length
    min_len = min(original_audio.shape[0], son_est.shape[0])
    mixed_est = original_audio[:min_len] + son_est[:min_len]
    mixed_ref = original_audio[:min_len] + son_ref[:min_len]

    # Normalize the mixed audio
    mixed_est /= np.max(np.abs(mixed_est))
    mixed_ref /= np.max(np.abs(mixed_ref))

    # Play the mixed audio
    display(Audio(mixed_est, rate=sr))
    display(Audio(mixed_ref, rate=sr))
