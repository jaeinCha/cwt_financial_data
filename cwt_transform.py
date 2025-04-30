"""
CWT implementation based on Torrence and Compo (1998).
"""

import numpy as np
from scipy.fftpack import fft, ifft
from wavelet_base import morlet_ft, mexican_hat_ft, paul_ft, cone_of_influence

def cwt(data, dt, scales, wavelet='morlet', omega0=6.0, m=4):
    """
    Continuous wavelet transform using the specified wavelet.
    
    Parameters:
    -----------
    data : array_like
        Time series data
    dt : float
        Time step
    scales : array_like
        Wavelet scales
    wavelet : str
        Wavelet type ('morlet', 'mexican_hat', or 'paul')
    omega0 : float
        Nondimensional frequency parameter for the Morlet wavelet
    m : int
        Order parameter for the Paul wavelet
        
    Returns:
    --------
    cwt : array_like
        Continuous wavelet transform (complex)
    """
    # Data preparation
    n = len(data)
    
    # Pad the data to the next power of 2
    padded_n = 2 ** int(np.ceil(np.log2(n)))
    padding = padded_n - n
    padded_data = np.zeros(padded_n)
    padded_data[:n] = data - np.mean(data)  # Remove mean
    
    # Compute FFT of the padded data
    fft_data = fft(padded_data)
    
    # Angular frequencies for the FFT
    omega_k = np.fft.fftfreq(padded_n, dt) * 2 * np.pi
    
    # Select wavelet function
    if wavelet == 'morlet':
        wavelet_ft = lambda s, omega: morlet_ft(omega, s, omega0)
    elif wavelet == 'mexican_hat':
        wavelet_ft = lambda s, omega: mexican_hat_ft(omega, s)
    elif wavelet == 'paul':
        wavelet_ft = lambda s, omega: paul_ft(omega, s, m)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet}")
    
    # Compute CWT for each scale
    cwt_result = np.zeros((len(scales), n), dtype=complex)
    
    for i, scale in enumerate(scales):
        # Compute wavelet transform for the current scale
        daughter_wavelet = wavelet_ft(scale, omega_k)
        
        # Multiply by FFT of the data
        transform = ifft(fft_data * daughter_wavelet)
        
        # Remove padding
        cwt_result[i, :] = transform[:n]
    
    return cwt_result

def cwt_power(W):
    """
    Compute the wavelet power spectrum.
    
    Parameters:
    -----------
    W : array_like
        Continuous wavelet transform (complex)
        
    Returns:
    --------
    power : array_like
        Wavelet power spectrum (|W|^2)
    """
    return np.abs(W)**2

def cwt_significance(power, scales, dt, alpha=0.05, lag1=0.0, fft_theor=None, dof=None):
    """
    Compute significance levels for wavelet power spectrum.
    
    Parameters:
    -----------
    power : array_like
        Wavelet power spectrum
    scales : array_like
        Wavelet scales
    dt : float
        Time step
    alpha : float
        Significance level (default=0.05)
    lag1 : float
        Lag-1 autocorrelation for red noise (default=0.0)
    fft_theor : array_like, optional
        Theoretical red noise spectrum
    dof : array_like, optional
        Degrees of freedom for significance tests
        
    Returns:
    --------
    signif : array_like
        Significance levels for the power spectrum
    """
    # Number of points
    n = power.shape[1]
    
    # Calculate background red noise spectrum if not provided
    if fft_theor is None:
        # Red noise background, following Torrence and Compo (1998)
        freq = np.fft.fftfreq(n, dt)
        freq[0] = 1e-6  # Avoid divide by zero
        fft_theor = (1 - lag1**2) / (1 - 2*lag1*np.cos(2*np.pi*freq*dt) + lag1**2)
    
    # Default DOF is 2 (real and imaginary parts of transform)
    if dof is None:
        dof = 2
    
    # Chi-square percent point function for the specified significance level
    from scipy.stats import chi2
    chisquare = chi2.ppf(1 - alpha, dof) / dof
    
    # Calculate significance levels
    signif = np.zeros((len(scales), n))
    for i in range(len(scales)):
        signif[i, :] = fft_theor * chisquare
    
    return signif

def smoothed_cwt_power(power, dt, scales, smoothing_radius=1, time_window=None):
    """
    Apply smoothing to the wavelet power spectrum in both time and scale.
    
    Parameters:
    -----------
    power : array_like
        Wavelet power spectrum
    dt : float
        Time step
    scales : array_like
        Wavelet scales
    smoothing_radius : int
        Radius for smoothing (default=1)
    time_window : int, optional
        Custom time averaging window length (overrides smoothing_radius)
        
    Returns:
    --------
    smoothed_power : array_like
        Smoothed wavelet power spectrum
    """
    # Get dimensions
    n_scales, n_times = power.shape
    
    # Initialize smoothed power
    smoothed_power = np.zeros_like(power)
    
    # Define time window for each scale (larger scales get larger windows)
    if time_window is None:
        time_window = np.zeros(n_scales, dtype=int)
        for i in range(n_scales):
            # Window size proportional to scale
            time_window[i] = max(1, int(scales[i] * smoothing_radius / dt))
    
    # Apply smoothing
    for i in range(n_scales):
        window = time_window[i]
        for j in range(n_times):
            # Time range for averaging
            j_start = max(0, j - window)
            j_end = min(n_times, j + window + 1)
            
            # Scale range for averaging (fixed)
            i_start = max(0, i - smoothing_radius)
            i_end = min(n_scales, i + smoothing_radius + 1)
            
            # Compute average
            patch = power[i_start:i_end, j_start:j_end]
            smoothed_power[i, j] = np.mean(patch)
    
    return smoothed_power

def scale_averaged_power(power, scales, dj):
    """
    Calculate scale-averaged wavelet power in a range of scales.
    
    Parameters:
    -----------
    power : array_like
        Wavelet power spectrum
    scales : array_like
        Wavelet scales
    dj : float
        Spacing between scales (in octaves)
        
    Returns:
    --------
    scale_avg : array_like
        Scale-averaged power over the specified range
    """
    # Constants from Torrence and Compo (1998)
    C_delta = 0.776  # For Morlet wavelet with omega0 = 6
    
    # Calculate scale-averaged power
    n_times = power.shape[1]
    scale_avg = np.zeros(n_times)
    
    for i in range(len(scales)):
        scale_avg += power[i, :] / scales[i]
    
    # Normalization
    scale_avg *= dj / C_delta
    
    return scale_avg

def compute_coi_mask(W, scales, coi):
    """
    Create a mask for the cone of influence.
    
    Parameters:
    -----------
    W : array_like
        Wavelet transform or power spectrum
    scales : array_like
        Wavelet scales
    coi : array_like
        Cone of influence
        
    Returns:
    --------
    mask : array_like
        Boolean mask, True for points outside COI
    """
    mask = np.ones_like(W, dtype=bool)
    
    n_scales, n_times = W.shape
    
    for i, scale in enumerate(scales):
        for j, c in enumerate(coi):
            if j < n_times:  # Make sure we're within bounds
                if scale > c:
                    if i < n_scales:  # Make sure we're within bounds
                        mask[i, j] = False
    
    return mask