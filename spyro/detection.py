import logging
import numpy as np

from spyro.utils import split_relaxed, validate_common_args
from spyro.utils import assert_alternating, assert_monotonic_increasing, assert_nonintersecting

logger = logging.getLogger("spyro")

__all__ = [
    "get_respiratory_extrema",
]


def get_respiratory_extrema(data, sfreq, *, device, window_sizes=None, decision_threshold="elbow"):
    """
    Estimate peaks and troughs of inhales and exhales.

    Accounts for ``species``-specific respiration rates using different window sizes and peak/trough
    interpretation across different ``device`` measurements [Noto2018]_.

    See Figure 2 of [Noto2018]_.

    1. Choose sliding window sizes.
    2. Choose sliding window offsets.
    3. Choose inhale/exhale thresholds (i.e., n_stds to be identified as extremum).
    4. Apply padding.
    5. Tally detected inhale/exhale extrema using each sliding window (and offsets).
    6. Identify PWCT algorithmically (optional).
    7. Apply PWCT to identify inhale/exhale extrema.
    8. Apply corrections to ensure alternating and paired inhales/exhales.
    9. Remove padding.

    .. note::
        The extrema (max/min, or peaks/troughs) are interpreted differently across devices.
        In respiration belt recordings, the peaks and troughs represent exhale and inhale onsets,
        respectively, because the point where volume has maximized and instantaneously decreases
        demarcates an exhale. This is unlike zero-crosses which demarcate breath onsets in airflow
        recordings. In airflow recordings, the extrema represent the peak flow rates of
        inhales/exhales. In rodent thermocouple recordings, the peaks and troughs represent inhale
        and exhale onsets, respectively. Inhales decrease the temperature in the nose and the onset
        is demarcated by the first point this happens - the inflection point of the peaks. Vice
        versa for exhales.

    .. note::

        * Always starts with inhale (peak) and ends on exhale (trough).
        * Inhales always occur before exhales (i.e., inhales[i] will always be < exhales[i]).
        * The final inhale is discarded if there is no subsequent exhale.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    device : str
        The device name of ``data``.
    window_sizes : list or :py:class:`numpy.ndarray` or None
        A sequence of integers to use for window sizes (TODO: explain more).
        Units are seconds.
        If a string, must be one of ``'human'`` or ``'rodent'``, and default window sizes are
        applied (see BreathMetrics [Noto2018]_).
    decision_threshold : int or str
        Determines the number of extrema required for a sample to be considered a peak or trough.
        TODO: Expand.
        AKA Percent Window Consensus Threshold (PWCT) in the paper?

    Returns
    -------
    inhales : :py:class:`numpy.ndarray`
        The location (sample index) of each inhalation peak.
    exhales : :py:class:`numpy.ndarray`
        The location (sample index) of each exhalation trough.

    References
    ----------
    .. [Noto2018] Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018).
                  Automated analysis of breathing waveforms using BreathMetrics: a respiratory
                  signal processing toolbox. Chemical Senses, 43(8), 583-597.
                  https://doi.org/10.1093/chemse/bjy045
    """
    # Q: Article says 300, 500, 700, 1000, and 5000
    #    Toolbox says 100, 300, 700, 1000, and 5000
    species_windows = {"human": [0.1, 0.3, 0.7, 1, 5], "rodent": [0.005, 0.010, 0.020, 0.050]}

    # Select window shifts.
    # Shifting/offsetting window to be unbiased by starting point.
    # TODO: I think windows get shifted and then averaged together?
    # TODO: Is this the same as the description in paper (ie, 33/66)?
    # TODO: Parameterize as keyword argument.
    offset_divisors = [1, 2, 3]

    # Validate input.
    validate_common_args(data=data, sfreq=sfreq, device=device)
    assert isinstance(window_sizes, (list, str)), "`window_sizes` must be a list or string"
    assert isinstance(decision_threshold, (int, np.integer, str)), "`decision_threshold` must be an integer or a string"
    if isinstance(decision_threshold, str):
        assert decision_threshold == "elbow", "`decision_threshold` must be an integer or 'elbow'"
    if isinstance(window_sizes, list):
        assert all(isinstance(x, (int, np.integer)) for x in window_sizes), "`window_sizes` must be a list of integers or a string" 
    elif isinstance(window_sizes, str):
        assert window_sizes in species_windows, (
            "`window_sizes` must be a list or a valid species name (%s)" %  ", ".join(species_windows)
        )
        window_sizes = species_windows[window_sizes]

    # Convert windows sizes from seconds to n_samples.
    # Q: I think window_size might be doubled for each iteration below?
    window_sizes = [ np.floor(sfreq * w).astype(int) for w in window_sizes ]

    ############################################################################
    # Find maxima/minima in multiple shifted version of each sliding window
    # and return peaks that are agreed upon by the majority of windows.
    ###### WRONG??????
    # Also each window gets shifted 3 times?
    # I think total n iterations is n_windows * n_offsets * n_slides?
    # But n_slides isn't added to total counts, just to verify original location?
    ############################################################################

    # Pad `data` with zeros so large windows can search the end of `data`.
    # TODO: I'm confused bc this doesn't appear to zero-pad like the comment suggests?
    #       It seems to add a reversed version of resp/data?
    n_samples = data.size
    largest_window = max(window_sizes)
    # Smallest between respiration size and the largest window
    # TODO: I think it's *2 because window size is mirrored?
    n_pad = min([n_samples, largest_window * 2])
    # TODO: change to np.pad(mode="reflect or symmetry")
    data_padded = np.concatenate([data, np.flip(data[-n_pad:])])
    # data_padded = np.pad(resp, (0, data.size))

    # Initialize a vector of zeros, holding total counts for each sample being identified as a peak or trough.
    # TODO: Consider a different approach, e.g., summing binary arrays.
    peak_counts = np.zeros_like(data_padded)
    trough_counts = np.zeros_like(data_padded)

    # Select thresholds for locating peak and troughs.
    # Peaks and troughs must exceed this value.
    # TODO: Parameterize as keyword argument.
    peak_threshold = data.mean() + data.std(ddof=1)
    trough_threshold = -1 * peak_threshold

    # In each sliding window (and each shift within sliding window), return peaks agreed upon by majority windows.
    # Find extrema in each window of the data using each `window_sizes` and `offset_divisors`.
    indices = np.arange(len(data_padded))
    for wsize in window_sizes:
        for oset in offset_divisors:
            # Shift starting point of sliding window to get unbiased maxes.
            # Iterate by this window size and find all maxima and minima.
            # Pick the start of the stream (will be 0, start of data, if shift/offset is 1).
            start_idx = (wsize - np.floor(wsize / oset).astype(int))
            # Split data into most possible chunks
            # TODO: If checking size, is custom function necessary? Or use np.array_split?
            # data_chunks = [ c for c in split_relaxed(indices[start_idx:], wsize) if c.size == wsize ]
            n_samples = len(lagged_start := indices[start_idx:])
            n_chunks, n_leftover = divmod(n_samples, wsize)
            chunks = np.array_split(lagged_start[:-n_leftover], n_chunks)
            # Check it there is a legitimate peak and/or trough in each chunk.
            for idx in chunks:
                # Extract data from this window.
                data_window = data_padded[idx]
                # Locate extrema and check if they pass predefined thresholds for peaks/troughs.
                # TODO: Probably need to make sure the extrema values are unique (esp bc of argmax/min).
                if data_window.max() > peak_threshold:
                    # Increase peaks counter at the location of this sample.
                    peak_counts[idx[data_window.argmax()]] += 1
                if data_window.min() < trough_threshold:
                    # Increase troughs counter at the location of this sample.
                    trough_counts[idx[data_window.argmin()]] += 1
    # peak_counts = (peak_counts > 0).astype(int)
    # peak_counts = (peak_counts > 0).astype(int)

    ############################################################################
    # DETERMINE NUMBER OF EXTREMA REQUIRED TO COUNT AS A PEAK/TROUGH
    ############################################################################

    if decision_threshold == "elbow":
        # Find threshold for *how many windows must agree* that makes minimal difference in number of extrema found
        # Similar to k-means elbow (or knee) method.
        # TODO: This is the number of windows that found a peak/trough?
        # TODO: How is this different than peak_counts?
        # - peak_counts has the number of times/windows *each sample* was picked
        # - n_peaks_found has the number of peaks found for *each window*?
        n_windows = len(window_sizes) * len(offset_divisors)
        # TODO: This should be plus 2?
        #       In the matlab code, I don't see how it would catch for the max amount of windows
        #       (wrongly zero but i guess nvr will one be found by ALL windows, impossible)
        # number of peaks that were found by more than `i` amount of window?
        n_peaks_found = [ np.sum(peak_counts > i) for i in range(1, n_windows + 1) ]
        n_troughs_found = [ np.sum(trough_counts > i) for i in range(1, n_windows + 1) ]
        best_n_windows_for_peak_diff = np.diff(n_peaks_found).argmax()
        best_n_windows_for_trough_diff = np.diff(n_troughs_found).argmax()
        # TODO: Does not account for multiple values of max?
        decision_threshold = np.floor(
            np.mean(
                [best_n_windows_for_peak_diff, best_n_windows_for_trough_diff]
            )
        )

    # Use `decision_threshold` to find all samples where a peak or trough was located.
    # Convert peaks/troughs to inhales/exhales.
    # Peaks usually represent inhales, but exhales when `device` is 'belt'.
    if device == "belt":
        inhales = np.where(trough_counts >= decision_threshold)[0]
        exhales = np.where(peak_counts >= decision_threshold)[0]
    else:
        inhales = np.where(peak_counts >= decision_threshold)[0]
        exhales = np.where(trough_counts >= decision_threshold)[0]

    ############################################################################
    # CORRECTIONS (probably pull to separate function)
    ############################################################################
    # - Remove previously-added padding.
    # - Ensure alternating inhales/exhales. (Sometimes multiple inhales or exhales are found in
    #   sequence), but there should always be inhale -> exhale -> inhale -> exhale -> ...
    # - Ensure first extrema is an inhale.
    # - Remove final inhale if not paired with an exhale.

    # Find the first inhale.
    # TODO: I think this just ensures that the first exhale is after the first inhale.
    #       Could use this instead?
    # Ensure the first event is an inhalation.
    exhales = exhales[np.where(exhales > inhales.min())]  # Might not be necessary.

    in_i = 0  # inhale index
    ex_i = 0  # exhale index
    # TODO: Maybe a more concise way of accomplishing this?
    #       - Drop while statement
    #       - Look for all potential values for each inhale/exhale at once (instead of iteratively)
    # Iterate over each inhale location and exhale location...
    while in_i < len(inhales) - 1 and ex_i < len(exhales) - 1:

        # Check for sequential inhales.
        sequential_inhales = True
        while sequential_inhales:
            # ...extract the sample number of `data` for inhale and exhale...
            in_curr = inhales[in_i]  # current inhale
            ex_curr = exhales[ex_i]  # current exhale
            in_next = inhales[in_i + 1]  # next inhale
            if in_next <= ex_curr:
                # If next inhale location is lower (ie, sooner) than current exhale location,
                # that means two consecutive inhales were identified and must be corrected.
                # Take the larger peak.
                # Note: Using absolute value bc in the case of respiration belt it will be troughs.
                if np.abs(data_padded[in_next]) > np.abs(data_padded[in_curr]):
                    # Replace current inhale with the next inhale if it's higher.
                    inhales[in_i] = in_next
                # Remove next inhale from inhales (if it was larger, it overwrote current).
                inhales = np.delete(inhales, in_i + 1)
                # % there still might be another peak to remove so go back and check
                # % again
            else:
                # If next inhale location is higher (ie, later) than current exhale location, move on.
                sequential_inhales = False

        # Check for sequential exhales (see comments for sequential inhales).
        sequential_exhales = True
        while sequential_exhales:
            ex_curr = exhales[ex_i]
            ex_next = exhales[ex_i + 1]
            in_next = inhales[in_i + 1]
            if ex_next <= in_next:
                if np.abs(data_padded[ex_next]) > np.abs(data_padded[ex_curr]):
                    exhales[ex_i] = ex_next
                exhales = np.delete(exhales, ex_i + 1)
            else:
                sequential_exhales = False

        # Make sure everything got corrected as-planned, with inhale before exhale.
        assert inhales[in_i] < exhales[ex_i], "The current inhale is unexpectedly greater than or equal to the current exhale."

        # Move indices forward.
        in_i += 1
        ex_i += 1

    # Remove previously-added padding.
    inhales = inhales[np.where(inhales < len(data))]
    exhales = exhales[np.where(exhales < len(data))]

    # Ensure no lonely inhale at the end.
    inhales = inhales[:len(exhales)]  # Might not be necessary.

    assert len(inhales) == len(exhales), "`inhales` and `exhales` must be the same length"
    assert_monotonic_increasing(inhales)
    assert_monotonic_increasing(exhales)
    assert_nonintersecting(inhales, exhales)
    assert_alternating(inhales, exhales)

    return inhales, exhales

