"""
Methods to reduce edge effects in wavelet analysis.
"""

import numpy as np
from scipy import signal

def extend_time_series(data, extension_type='reflection', extension_length=None):
    """
    Extend a time series to reduce edge effects in wavelet analysis.
    
    Parameters:
    -----------
    data : array_like
        Time series data
    extension_type : str
        Method for extending the data:
        - 'reflection': Mirror the data at the endpoints
        - 'periodic': Assume the data is periodic
        - 'symmetric': Symmetric extension
        - 'ar_model': Use an AR model to predict extension
        - 'mean_padded': Pad with the mean of the data
    extension_length : int, optional
        Length of the extension on each side. If None, uses max(len(data)//4, 20)
        
    Returns:
    --------
    extended_data : array_like
        Extended time series
    extension_indices : tuple
        Indices (start, end) of the original data in the extended series
    """
    if extension_length is None:
        extension_length = max(len(data) // 4, 20)
    
    n = len(data)
    
    if extension_type == 'reflection':
        # Mirror the data at the endpoints
        left_extension = data[1:extension_length+1][::-1]
        right_extension = data[-extension_length-1:-1][::-1]
        extended_data = np.concatenate([left_extension, data, right_extension])
    
    elif extension_type == 'periodic':
        # Assume the data is periodic
        left_extension = data[-extension_length:]
        right_extension = data[:extension_length]
        extended_data = np.concatenate([left_extension, data, right_extension])
    
    elif extension_type == 'symmetric':
        # Symmetric extension
        left_extension = data[:extension_length][::-1]
        right_extension = data[-extension_length:][::-1]
        extended_data = np.concatenate([left_extension, data, right_extension])
    
    elif extension_type == 'ar_model':
        # Use an AR model to predict extension
        # Fit AR model on the data
        ar_order = min(50, n // 10)  # Reasonable AR order
        ar_model = signal.ar_covar(data, ar_order)[0]
        
        # Generate left extension (backward prediction)
        left_extension = np.zeros(extension_length)
        reversed_data = data[:ar_order][::-1]
        for i in range(extension_length):
            pred = np.sum(ar_model * np.roll(np.append(reversed_data, left_extension[:i]), 1)[:ar_order])
            left_extension[i] = pred
        left_extension = left_extension[::-1]  # Reverse back
        
        # Generate right extension (forward prediction)
        right_extension = np.zeros(extension_length)
        last_values = data[-ar_order:]
        for i in range(extension_length):
            pred = np.sum(ar_model * np.append(last_values[-(ar_order-1):], right_extension[:i]))
            right_extension[i] = pred
            
        extended_data = np.concatenate([left_extension, data, right_extension])
    
    elif extension_type == 'mean_padded':
        # Pad with the mean of the data
        mean_value = np.mean(data)
        left_extension = np.ones(extension_length) * mean_value
        right_extension = np.ones(extension_length) * mean_value
        extended_data = np.concatenate([left_extension, data, right_extension])
    
    else:
        raise ValueError(f"Unknown extension type: {extension_type}")
    
    extension_indices = (extension_length, extension_length + n)
    
    return extended_data, extension_indices

def process_extended_results(wavelet_result, extension_indices):
    """
    Extract the original portion of the wavelet transform from the extended result.
    
    Parameters:
    -----------
    wavelet_result : array_like
        Wavelet transform of the extended data
    extension_indices : tuple
        Indices (start, end) of the original data in the extended series
        
    Returns:
    --------
    original_result : array_like
        Wavelet transform corresponding to the original data
    """
    start, end = extension_indices
    if wavelet_result.ndim == 2:
        # For 2D results like CWT
        return wavelet_result[:, start:end]
    else:
        # For 1D results
        return wavelet_result[start:end]

def taper_time_series(data, taper_length=None, taper_type='cosine'):
    """
    Apply a taper (window function) to the ends of the time series to reduce edge effects.
    
    Parameters:
    -----------
    data : array_like
        Time series data
    taper_length : int, optional
        Length of the taper on each side. If None, uses max(len(data)//10, 10)
    taper_type : str
        Type of tapering window:
        - 'cosine': Cosine taper
        - 'hann': Hann window
        - 'hamming': Hamming window
        
    Returns:
    --------
    tapered_data : array_like
        Tapered time series
    """
    n = len(data)
    
    if taper_length is None:
        taper_length = max(n // 10, 10)
    
    # Create appropriate window
    if taper_type == 'cosine':
        taper = np.cos(np.linspace(np.pi/2, 0, taper_length))**2
    elif taper_type == 'hann':
        taper = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_length)))
    elif taper_type == 'hamming':
        taper = 0.54 - 0.46 * np.cos(np.pi * np.linspace(0, 1, taper_length))
    else:
        raise ValueError(f"Unknown taper type: {taper_type}")
    
    # Create a window of ones with tapered ends
    window = np.ones(n)
    window[:taper_length] = taper
    window[-taper_length:] = taper[::-1]
    
    # Apply the window
    tapered_data = data * window
    
    return tapered_data

def boundary_aware_coi(n, dt, scales, wavelet='morlet', omega0=6.0):
    """
    Calculate a more conservative cone of influence that takes into account
    the boundary handling method used.
    
    Parameters:
    -----------
    n : int
        Length of the time series
    dt : float
        Time step
    scales : array_like
        Wavelet scales
    wavelet : str
        Wavelet type
    omega0 : float
        Nondimensional frequency parameter for the Morlet wavelet
        
    Returns:
    --------
    coi : array_like
        Cone of influence values for each time step
    """
    # Calculate e-folding time for the wavelet
    if wavelet == 'morlet':
        const = np.sqrt(2) * 2  # More conservative than the standard sqrt(2)
    elif wavelet == 'paul':
        const = 2 / np.sqrt(2)
    elif wavelet in ['mexican_hat', 'dog']:
        const = np.sqrt(2) * 2
    else:
        const = np.sqrt(2)
    
    # Calculate COI at each time step with a more conservative approach
    coi = np.zeros(n)
    half_len = (n - 1) // 2
    
    for i in range(n):
        if i <= half_len:
            dist_from_edge = i
        else:
            dist_from_edge = n - 1 - i
        
        # More restrictive COI that treats coefficients near the boundary with more caution
        coi[i] = dt * (dist_from_edge + 1) * const
    
    return coi