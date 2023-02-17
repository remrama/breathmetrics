"""
Preprocessing functions (i.e., cleaning/de-noising/standardizing respiratory time series).
"""
import logging
import numpy as np
from scipy import signal, stats

from spyro.utils import moving_average, validate_common_args


logger = logging.getLogger("spyro")


def smooth(data, sfreq, window_size):
    """Smooth (denoise) respiratory data by applying a moving average to the signal.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    window_size : int or str
        The window size of the moving average. If a string, must be one of ``'human'`` or
        ``'rodent'``, and default window sizes are applied (see BreathMetrics [Noto2018]_).

    References
    ----------
    .. [Noto2018] Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018).
                  Automated analysis of breathing waveforms using BreathMetrics: a respiratory
                  signal processing toolbox. Chemical Senses, 43(8), 583-597.
                  https://doi.org/10.1093/chemse/bjy045
    """
    validate_common_args(data=data, sfreq=sfreq)
    assert isinstance(window_size, (int, str)), "`window_size` must be an integer or string"
    species_windows = {"human": 50, "rodent": 10}
    if isinstance(window_size, str):
        assert window_size in species_windows, (
            "`window_size` must be an integer or a valid species name (%s)" %  ", ".join(species_windows)
        )
        # Get the integer value window size for this species.
        window_size = species_windows[window_size]
    # TODO: Why is this divied by 1000 but the other one isn't? Are these window sizes not in seconds?
    sfreq_corrected_window_size = np.floor(sfreq / 1000 * window_size).astype(int)
    data_smooth = moving_average(data, sfreq_corrected_window_size)
    # TODO: Maybe use this instead?
    # from scipy.signal import savgol_filter
    # yhat = savgol_filter(y, 51, 3)  # window size 51, polynomial order 3
    return data_smooth


def correct_baseline(data, sfreq, window_size=60, zscore=False):
    """
    Correct respiratory traces by (1) removing linear trends, (2) removing offsets,
    and (3) normalizing raw values.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    window_size : int or None
        If None, remove offsets by subtracting the global mean signal.
        If an integer, remove offsets by subtracting the mean signal calculated over a sliding
        window of ``window_size`` seconds(?!) (i.e., moving average).

        .. warning:: ``window_size=None`` is much faster, but does not remove acute drifts.

    zscore : bool
        If True, z-score the detrended and demeaned data. 
    """
    validate_common_args(data=data, sfreq=sfreq)
    assert isinstance(window_size, (type(None), int)), "`window_size` must be None or an integer"
    assert isinstance(zscore, bool), "`zscore` must be True or False"
    # TODO: Check the order of operations here against original. (both global and acute corrections?)
    # Remove global linear trends.
    # TODO: Should this use be using the "choice_resp" thing?
    corrected = signal.detrend(data, type="linear")
    # Remove offsets.
    if window_size is None:
        corrected = signal.detrend(corrected, type="constant")
    else:
        # Remove periodic drifts.
        # TODO: Can this be replaced with the `bp` argument in signal.detrend?
        sfreq_corrected_smoothing_window = np.floor(sfreq * window_size).astype(int)
        logger.info("Calculating %i-sec window sliding average", window_size)
        sliding_mean = moving_average(corrected, sfreq_corrected_smoothing_window)
        # Subtract sliding mean from respiratory trace.
        corrected = corrected - sliding_mean
    # Normalize amplitudes.
    if zscore:
        corrected = stats.zscore(corrected, ddof=0)
    return corrected
