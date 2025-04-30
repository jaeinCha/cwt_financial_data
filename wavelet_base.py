"""
Core wavelet functions for CWT analysis.
Based on "A Practical Guide to Wavelet Analysis" by Torrence and Compo (1998).
"""

import numpy as np
from scipy import signal

def morlet(t, omega0=6.0):
    """
    Morlet wavelet function.
    
    Parameters:
    -----------
    t : array_like
        Time array
    omega0 : float
        Nondimensional frequency parameter (default=6.0)
        
    Returns:
    --------
    wavelet : array_like
        Morlet wavelet at times t
    """
    return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)

def morlet_ft(omega, s, omega0=6.0):
    """
    Fourier transform of the Morlet wavelet.
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array
    s : float
        Scale parameter
    omega0 : float
        Nondimensional frequency parameter (default=6.0)
        
    Returns:
    --------
    hat : array_like
        Fourier transform of the Morlet wavelet
    """
    H = np.ones(len(omega))  # Heaviside step function
    H[omega <= 0] = 0
    
    # Fourier transform of Morlet wavelet
    return np.pi**(-0.25) * H * np.exp(-(s*omega - omega0)**2 / 2)

def mexican_hat(t):
    """
    Mexican hat (Ricker) wavelet function.
    
    Parameters:
    -----------
    t : array_like
        Time array
        
    Returns:
    --------
    wavelet : array_like
        Mexican hat wavelet at times t
    """
    return (2/np.sqrt(3)) * np.pi**(-0.25) * (1 - t**2) * np.exp(-t**2/2)

def mexican_hat_ft(omega, s):
    """
    Fourier transform of the Mexican hat wavelet.
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array
    s : float
        Scale parameter
        
    Returns:
    --------
    hat : array_like
        Fourier transform of the Mexican hat wavelet
    """
    return np.sqrt(8/3) * np.pi**(-0.25) * s**2 * omega**2 * np.exp(-s**2 * omega**2 / 2)

def paul(t, m=4):
    """
    Paul wavelet function.
    
    Parameters:
    -----------
    t : array_like
        Time array
    m : int
        Order parameter (default=4)
        
    Returns:
    --------
    wavelet : array_like
        Paul wavelet at times t
    """
    const = (2**m * np.math.factorial(m) / np.sqrt(np.pi * (2*m+1)))
    return const * (1 - 1j*t)**(-(m+1))

def paul_ft(omega, s, m=4):
    """
    Fourier transform of the Paul wavelet.
    
    Parameters:
    -----------
    omega : array_like
        Angular frequency array
    s : float
        Scale parameter
    m : int
        Order parameter (default=4)
        
    Returns:
    --------
    hat : array_like
        Fourier transform of the Paul wavelet
    """
    H = np.ones(len(omega))  # Heaviside step function
    H[omega <= 0] = 0
    
    const = 2**m / np.sqrt(m * (2*m-1))
    return const * H * (s * omega)**m * np.exp(-s * omega)

def fourier_period(s, wavelet='morlet', omega0=6.0, m=4):
    """
    Convert wavelet scale to equivalent Fourier period.
    
    Parameters:
    -----------
    s : float
        Wavelet scale
    wavelet : str
        Wavelet type ('morlet', 'mexican_hat', or 'paul')
    omega0 : float
        Nondimensional frequency parameter for the Morlet wavelet
    m : int
        Order parameter for the Paul wavelet
        
    Returns:
    --------
    period : float
        Equivalent Fourier period
    """
    if wavelet == 'morlet':
        return 4 * np.pi * s / (omega0 + np.sqrt(2 + omega0**2))
    elif wavelet == 'mexican_hat':
        return 2 * np.pi * s / np.sqrt(2.5)
    elif wavelet == 'paul':
        return 4 * np.pi * s / (2*m + 1)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet}")

def scale_from_period(period, wavelet='morlet', omega0=6.0, m=4):
    """
    Convert Fourier period to wavelet scale.
    
    Parameters:
    -----------
    period : float
        Fourier period
    wavelet : str
        Wavelet type ('morlet', 'mexican_hat', or 'paul')
    omega0 : float
        Nondimensional frequency parameter for the Morlet wavelet
    m : int
        Order parameter for the Paul wavelet
        
    Returns:
    --------
    s : float
        Equivalent wavelet scale
    """
    if wavelet == 'morlet':
        return (period * (omega0 + np.sqrt(2 + omega0**2))) / (4 * np.pi)
    elif wavelet == 'mexican_hat':
        return period * np.sqrt(2.5) / (2 * np.pi)
    elif wavelet == 'paul':
        return period * (2*m + 1) / (4 * np.pi)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet}")

def cone_of_influence(N, dt, scales, wavelet='morlet', omega0=6.0, m=4):
    """
    Calculate the cone of influence (COI) for the given wavelet.
    Points inside the COI are affected by edge effects.
    
    Parameters:
    -----------
    N : int
        Number of time steps
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
    coi : array_like
        Cone of influence for each time step
    """
    # e-folding time
    if wavelet == 'morlet':
        coi_const = np.sqrt(2)
    elif wavelet == 'paul':
        coi_const = 1 / np.sqrt(2)
    elif wavelet == 'mexican_hat':
        coi_const = np.sqrt(2)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet}")
    
    # Calculate COI at each time step
    coi = np.zeros(N)
    half_len = (N - 1) // 2
    
    for i in range(N):
        if i <= half_len:
            coi[i] = dt * i * coi_const if i > 0 else dt * coi_const
        else:
            coi[i] = dt * (N - 1 - i) * coi_const
    
    return coi