"""
Preprocessing functions (i.e., cleaning/de-noising/standardizing respiratory time series).
"""
import logging
import numpy as np
from scipy import signal, stats

from spyro.utils import moving_average, validate_common_args


logger = logging.getLogger("spyro")

__all__ = [
    "clean_signal",
]


def clean_signal(data, sfreq, pad_mode="reflect", smoothing_window="human", detrending_window=60, zscore=False):
    """
    Correct/denoise respiratory traces by:
    1. Applying a moving average to the signal.
    2. Removing linear trends.
    3. Removing offsets
    4. Normalizing raw values.

    See "Data normalization and smoothing" subheading in [Noto2018]_.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    pad_mode : str
        Pad mode for :py:func:`numpy.pad`.
    smoothing_window : int or str
        The window size of the moving average in seconds. If a string, must be one of ``'human'`` or
        ``'rodent'``, and default window sizes are applied (0.25 s for human, [Noto2018]_).
    detrending_window : int or None
        If None, remove offsets by subtracting the global mean signal.
        If an integer, remove offsets by subtracting the mean signal calculated over a sliding
        window of ``window_size`` seconds(?!) (i.e., moving average).
        Default value (60 seconds) is recommended in [Noto2018]_.

        .. warning:: ``window_size=None`` is much faster, but does not remove acute drifts.
    zscore : bool
        If True, z-score the detrended and demeaned data. 

    References
    ----------
    .. [Noto2018] Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018).
                  Automated analysis of breathing waveforms using BreathMetrics: a respiratory
                  signal processing toolbox. Chemical Senses, 43(8), 583-597.
                  https://doi.org/10.1093/chemse/bjy045
    """
    default_smoothing_windows = {"human": 0.025, "rodent": 0.005}

    validate_common_args(data=data, sfreq=sfreq)
    assert isinstance(pad_mode, str), "`pad_mode` must be a string"
    assert isinstance(smoothing_window, (int, str)), "`window_size` must be an integer or string"
    # TODO: Check converted rodent value is okay.
    if isinstance(smoothing_window, str):
        assert smoothing_window in default_smoothing_windows, (
            "`window_size` must be an integer or a valid species name (%s)" %  ", ".join(default_smoothing_windows)
        )
        # Get the integer value window size for this species.
        smoothing_window = default_smoothing_windows[smoothing_window]
    assert isinstance(detrending_window, (int, type(None))), "`window_size` must be None or an integer"
    assert isinstance(zscore, bool), "`zscore` must be True or False"

    # Convert window sizes from seconds to number of samples.
    smoothing_window_in_samples = np.floor(sfreq * smoothing_window).astype(int)
    if detrending_window is not None:
        detrending_window_in_samples = np.floor(sfreq * detrending_window).astype(int)

    # Pad data.
    cleaned = np.pad(data, smoothing_window_in_samples, mode=pad_mode)
    
    # Smooth data.
    cleaned = moving_average(cleaned, smoothing_window_in_samples)
    # TODO: Maybe use this to smooth instead? scipy.signal.savgol_filter(y, 51, 3)  # window size 51, polynomial order 3

    # Remove global linear trends.
    cleaned = signal.detrend(cleaned, type="linear")
    # Remove offsets.
    if detrending_window is None:
        cleaned = signal.detrend(cleaned, type="constant")
    else:
        # Remove periodic drifts.
        # Subtract sliding mean from respiratory trace.
        logger.info("Calculating %i-sec window sliding average", detrending_window)
        cleaned -= moving_average(cleaned, detrending_window_in_samples)
    # Normalize amplitudes.
    if zscore:
        cleaned = stats.zscore(cleaned, ddof=0)

    # Remove padding.
    cleaned = cleaned[smoothing_window_in_samples:-smoothing_window_in_samples]

    return cleaned
