import mne
from mne import Epochs
from mne.io import BaseRaw, BaseRaw

############################ Preprocessing functions #############################################
def pick_channels(raw, channels = None , **kwargs):
    """
    Pick channels from raw data.

    Args:
        raw (mne.io.BaseRaw): The raw data.
        channels (list): The list of channels to pick.

    Returns:
        mne.io.Raw: The raw data with the selected channels.
    """
    # Check if raw is an instance of mne.io.Raw+
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    
    # Here we need to find the stim_channels and then pick the channels
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)     
    
    if channels is None:
        return raw
    else:
        if len(stim_channels) > 0:
            if isinstance(channels, list):
                channels = channels + stim_channels
        else:
            channels = channels + ['stim']
            
        return  raw.copy().pick_channels(ch_names = channels, **kwargs)

def filter_raw(raw, l_freq, h_freq, **kwargs):
    """
    Filters the raw data using a bandpass filter.

    Parameters:
    raw (mne.io.Raw): The raw data to be filtered.
    l_freq (float): The lower frequency of the bandpass filter.
    h_freq (float): The higher frequency of the bandpass filter.
    **kwargs: Additional keyword arguments to be passed to the filter method.

    Returns:
    mne.io.Raw: The filtered raw data.
    """
    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    
    return raw.filter(l_freq=l_freq, h_freq=h_freq, **kwargs)

def filter_epoch(epoch, l_freq, h_freq, **kwargs):
    """
    Filters the epoched EEG data using a bandpass filter.

    Parameters:
    epoch (mne.Epochs): The epoch to be resampled.
    l_freq (float): The lower frequency of the bandpass filter.
    h_freq (float): The higher frequency of the bandpass filter.
    **kwargs: Additional keyword arguments to be passed to the filter method.

    Returns:
    mne.Epochs: The filtered raw data.
    """
    # Check if raw is an instance of mne.io.Raw
    if not isinstance(epoch, Epochs):
        raise TypeError("epoch is not an instance of mne.Epochs.")
    
    return epoch.filter(l_freq=l_freq, h_freq=h_freq, **kwargs)

def filter_notch_raw(raw, freqs, **kwargs):
    """
    Applies a notch filter to remove specific frequencies from the raw data.

    Parameters:
    raw (mne.io.Raw): The raw data to be filtered.
    freqs (list): The frequencies to be removed.
    **kwargs: Additional keyword arguments to be passed to the notch_filter method.

    Returns:
    mne.io.Raw: The filtered raw data.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    return raw.notch_filter(freqs=freqs, **kwargs)

def re_referencing_raw(raw, ref_channels, **kwargs):
    """
    Re-references the raw EEG data using the specified reference channels.

    Parameters:
    raw (mne.io.Raw): The raw EEG data.
    ref_channels (list): The reference channels to use for re-referencing.
    **kwargs: Additional keyword arguments to be passed to mne.set_eeg_reference().

    Returns:
    raw (mne.io.Raw): The re-referenced raw EEG data.
    ref_data (ndarray): The reference data used for re-referencing.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    raw  =  raw.set_eeg_reference(ref_channels, **kwargs)
    return raw

def re_referencing_epoch(epoch, ref_channels, **kwargs):
    """
    Re-references the epoched EEG data using the specified reference channels.

    Parameters:
    epoch (mne.Epochs): The epoch to be resampled.
    ref_channels (list): The reference channels to use for re-referencing.
    **kwargs: Additional keyword arguments to be passed to mne.set_eeg_reference().

    Returns:
    epoch (mne.Epochs): The re-referenced epoched EEG data.
    """
    if not isinstance(epoch, Epochs):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    return epoch.set_eeg_reference(ref_channels, **kwargs)

def resample_epoch(epoch, sfreq, **kwargs):
    """
    Resamples an epoch using the specified sampling frequency.

    Parameters:
    epoch (mne.Epochs): The epoch to be resampled.
    sfreq (float): The desired sampling frequency.
    **kwargs: Additional keyword arguments to be passed to the resample function.

    Returns:
    mne.Epochs: The resampled epoch.
    """
    if not isinstance(epoch, Epochs):
        raise TypeError("epoch is not an instance of mne.Epochs.")
    return epoch.resample(sfreq, **kwargs)

def convert_to_epochs(raw, events, event_id, tmin, tmax, **kwargs):
    """
    Converts the raw data to epochs using the specified events and event_id.

    Parameters:
    raw (mne.io.Raw): The raw data to be converted to epochs.
    events (ndarray): The events array.
    event_id (dict): The event id dictionary.
    tmin (float): The start time of the epoch.
    tmax (float): The end time of the epoch.
    **kwargs: Additional keyword arguments to be passed to the mne.Epochs() constructor.

    Returns:
    mne.Epochs: The epoch data.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw is not an instance of mne.io.Raw.")
    if 'preload' not in kwargs or not kwargs['preload']:
        kwargs['preload'] = True
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        **kwargs
    )

def crop_epoch(epoch, tmin, tmax, **kwargs):
    """
    Crop the given epoch data between the specified time points.

    Parameters:
        epoch (Epochs): The epoch data to be cropped.
        tmin (float): The start time point for cropping.
        tmax (float): The end time point for cropping.
        **kwargs: Additional keyword arguments to be passed to the crop method.

    Returns:
        Epochs: The cropped epoch data.

    """
    if isinstance(epoch, Epochs):
        return epoch.crop(tmin=tmin, tmax=tmax, **kwargs)

def convert_to_np_array(data):
    """
    Converts data to a NumPy array.

    Parameters:
    data (Epochs or BaseRaw): The data to be converted.

    Returns:
    numpy.ndarray: The converted data as a NumPy array.
    """
    if isinstance(data, Epochs) or isinstance(data, BaseRaw):
        return data.get_data(copy=True)

def apply_unit_factor(data, unit_factor):
    """
    Applies a unit factor to the data.

    Args:
        data (array-like): The data to apply the unit factor to.
        unit_factor (float): The unit factor to apply.

    Returns:
        array-like: The data multiplied by the unit factor.
    """
    return data * unit_factor


# AVAILABLE PIPELINES
import inspect
AVAILABLE_PIPELINES = []
# Crear una copia del diccionario globals()
globals_copy = dict(globals())
# Recorre todas las variables globales en la copia
for name, obj in globals_copy.items():
    # Si el objeto es una función, agrega su nombre a la lista
    if inspect.isfunction(obj):
        AVAILABLE_PIPELINES.append(name)