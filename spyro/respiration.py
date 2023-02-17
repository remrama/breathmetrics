"""
Main Respiration class.
"""
import logging
import numpy as np

from spyro.detection import get_inhales_exhales
from spyro.preprocessing import smooth, correct_baseline
from spyro.utils import moving_average, validate_common_args

logger = logging.getLogger("spyro")

__all__ = [
    "Respiration",
]


class Respiration:
    """
    Analyze and visualize several features of a respiratory trace. See [Noto2018]_ for details.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Respiration data, a numpy array of shape (1,).
    sfreq : float, int
        Sampling frequency of ``data`` in Hz.
    species : str
        The species of ``data``. Must be ``'human'`` (default) or ``'rodent'``.
    device : str
        The device name of ``data``. Must be one of the following:

        * ``'airflow'`` : Nasal(?) airflow recordings (default) 
        * ``'belt'`` : Respiration belt (only valid if ``species='human'``)
        * ``'thermocouple'`` : thermoelectrical thermometer(?) (only valid if ``species='rodent'``)

    References
    ----------
    .. [Noto2018] Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018).
                  Automated analysis of breathing waveforms using BreathMetrics: a respiratory
                  signal processing toolbox. Chemical Senses, 43(8), 583-597.
                  https://doi.org/10.1093/chemse/bjy045
 
    Examples
    --------
    >>> import numpy as np
    >>> import spyro
    >>> data = np.load("sample_data.npz")
    >>> signal, sfreq = data["resp"], data["sfreq"][0]
    >>> resp = spyro.Respiration(signal, sfreq=sfreq, species="human", device="airflow")
    """
    def __init__(self, data, sfreq, *, species="human", device="airflow"):
        # TODO: Ensure all functions take shape (N,) instead of (1, N)
        # TODO: Ensure everything indexes from 0 and not 1

        validate_common_args(data=data, sfreq=sfreq, species=species, device=device)
        if device in ["belt", "thermocouple"]:
            logger.warning("Only certain features can be derived from `%s` data.", device)

        # % time is like fieldtrip. All events are indexed by point in time vector.
        # get array of timepoints for each sample (IN SECONDS)
        time = np.arange(len(data)) / sfreq

        # Signal preprocessing.
        data_smooth = smooth(data, sfreq, window_size=species)
        data_corrected = correct_baseline(data_smooth, sfreq, window_size=60, zscore=False)

        # Feature detection.
        inhales, exhales = get_inhales_exhales(
            data=data_corrected, sfreq=sfreq, device=device, window_sizes=species, decision_threshold="elbow",
        )

        # Set attributes.
        self._sfreq = sfreq
        self._species = species
        self._device = device
        self._time = time
        self._data_raw = data
        self._data_smooth = data_smooth
        self._data_corrected = data_corrected
        self._inhales = inhales
        self._exhales = exhales

    @property
    def data(self):
        """The processed respiratory data."""
        return self._data_corrected

    @property
    def device(self):
        """Name of the device used for data collection."""
        return self._device

    @property
    def species(self):
        """Name of the species observed for data collection."""
        return self._species

    @property
    def sfreq(self):
        """The sampling frequency of data in Hertz (Hz)."""
        return self._sfreq

    @property
    def time(self):
        """An array indicating the time (in seconds) of each sample."""
        return self._time

    @property
    def inhales(self):
        """The index (i.e., sample number) of inhalation peaks."""
        return self._inhales

    @property
    def exhales(self):
        """The index (i.e., sample number) of exhalation troughs."""
        return self._exhales
