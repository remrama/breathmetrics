"""
Helper functions.
"""
import logging
import numpy as np


def set_log_level(verbose):
    """
    Parameters
    ----------
    verbose : str
        The verbosity of logging/printing. Must be one of debug, info, warning, error, or critical.
    """
    log_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    assert isinstance(verbose, str), "`verbose` must be a string"
    verbose = verbose.lower()
    assert verbose in log_levels, "`verbose` must one of %s" % ", ".join(log_levels)
    logger = logging.getLogger("spyro")
    level = log_levels[verbose]
    logger.setLevel(level)


def validate_common_args(*, data=None, sfreq=None, species=None, device=None):
    """Convenience function for checking input to various spyro functions. Raises error as needed.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    species : str
        The species of ``data``. Must be ``'human'`` or ``'rodent'``.
    device : str
        The device name of ``data``. Must be ``'airflow'``, ``'belt'``, or ``'thermocouple'``.
    """
    min_sfreq = 20
    max_sfreq = 5000
    valid_species = ["human", "rodent"]
    valid_devices = ["airflow", "belt", "thermocouple"]

    if data is not None:
        assert isinstance(data, np.ndarray), "`data` must be a numpy array"
        assert data.ndim == 1, "`data` must be a 1-dimensional numpy array"
        assert np.issubdtype(data.dtype, np.number), "`data` must contain numbers"
        assert np.all(np.isfinite(data)), "`data` must not contain NaN or infinity values"
    if sfreq is not None:
        assert isinstance(sfreq, (int, np.integer)), "`sfreq` must be an integer"
        assert min_sfreq <= sfreq <= max_sfreq, f"`sfreq` must be between {min_sfreq} and {max_sfreq}, inclusive"
    if species is not None:
        assert isinstance(species, str), "`species` must be a string"
        assert species in valid_species, f"`species` must be one of {valid_species}"
    if device is not None:
        assert device in valid_devices, f"`device` must be one of {valid_devices}"
        assert isinstance(device, str), "`device` must be a string"
    if species is not None and device is not None:
        assert not (species == "rodent" and device == "belt"), "`device` can't be belt if `species` is rodent"
        assert not (species == "human" and device == "thermocouple"), "`device` can't be thermocouple if `species` is human"


def moving_average(x, w):
    """Return a moving average.

    Parameters
    ----------
    x : :py:class:`numpy.ndarray`
        Data to apply moving average.
    w : int
        Window size in n_samples.

    Returns
    -------
    arr : :py:class:`numpy.ndarray`
        Numpy array of same size as ``x``, moving average applied.
    """
    assert isinstance(x, np.ndarray), "`x` must be a numpy array"
    assert isinstance(w, (int, np.integer)), "`w` must be an integer"
    return np.convolve(x, np.ones(w), mode="same") / w

# TODO: Compare with these other potential implementations for moving_average (and consider numba).
# def moving_average(x, w):
#     return signal.fftconvolve(x, np.ones(w), mode="same") / w
# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# def moving_average(x, w):
#     t = pd.date_range(w, freq="s")
#     ser = pd.Series(x, index=w)
#     # ser = ser.fillna?
#     ser.rolling_mean("s", 10)
#     return ser.to_numpy()
# This method might be closer to the BreathMetrics implementation: https://stackoverflow.com/a/20619164

def split_relaxed(lst, n):
    """Yield successive n-sized chunks from lst.

    Similar to :py:func:`numpy.array_split` but returns the final chunk even if incomplete.

    Parameters
    ----------
    lst : list or :py:class:`numpy.ndarray`
    n : int

    Returns
    -------
    """
    assert isinstance(lst, (list, np.ndarray)), "`lst` must be a list or numpy array"
    assert isinstance(n, (int, np.integer)), "`n` must be an integer"
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def assert_alternating(lst_a, lst_b):
    """Return True if
    Parameters
    ----------
    lst_a : list
    lst_b : list
    """
    assert isinstance(lst_a, (list, np.ndarray)), "`lst_a` must be a list or numpy array"
    assert isinstance(lst_b, (list, np.ndarray)), "`lst_b` must be a list or numpy array"
    assert len(lst_a) == len(lst_b), "`lst_a` and `lst_b` must be the same length"
    # assert all(b > a for a, b in zip(lst_a, lst_b)), "`lst_a` and `lst_b` must have alternating values"
    # assert all(b < a for a, b in zip(lst_a[1:], lst_b[:-1])), "`lst_a` and `lst_b` must have alternating values"
    assert np.all(lst_b > lst_a), "Each element of `lst_b` must be higher than its corresponding element in `lst_a`"
    assert np.all(lst_b[:-1] < lst_a[1:]), "Each element of `lst_b` must be lower than its prior element in `lst_a`"
    # TODO: Check speed for these different implementations.

def assert_nonintersecting(lst_a, lst_b):
    """Raise error if `lst_a` and `lst_b` contain any overlapping values.

    Parameters
    ----------
    lst_a : list
    lst_b : list

    """
    assert isinstance(lst_a, (list, np.ndarray)), "`lst_a` must be a list or numpy array"
    assert isinstance(lst_b, (list, np.ndarray)), "`lst_b` must be a list or numpy array"
    assert len(lst_a) == len(lst_b), "`lst_a` and `lst_b` must be the same length"
    assert not set(lst_a).intersection(set(lst_b)), "`lst_a` and `lst_b` must be unique"

def assert_monotonic_increasing(lst):
    """Raises error if `lst` is not strictly increasing.

    Parameters
    ----------
    lst : list
    """
    assert isinstance(lst, (list, np.ndarray)), "`lst` must be a list or numpy array"
    assert all(x < y for x, y in zip(lst, lst[1:])), "`lst` must be monotonic increasing"
    # TODO: Check speed against np.all(np.diff(lst) > 0)
