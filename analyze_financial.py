"""
Financial time series analysis using CWT.
Applies the methods from Eliasson's "An Application of the Continuous Wavelet Transform to Financial Time Series".
"""

import numpy as np
import pandas as pd
from scipy import stats
from cwt_transform import cwt, cwt_power, cwt_significance, smoothed_cwt_power, compute_coi_mask#, scale_averaged_power
from wavelet_base import fourier_period, scale_from_period, cone_of_influence

class FinancialWaveletAnalyzer:
    """
    Class for analyzing financial time series using wavelets.
    """
    
    def __init__(self, data, dt=1.0, wavelet='morlet', omega0=6.0, m=4):
        """
        Initialize the analyzer with time series data.
        
        Parameters:
        -----------
        data : array_like
            Financial time series data
        dt : float
            Time step (e.g., 1 for daily data, 1/252 for trading days)
        wavelet : str
            Wavelet type ('morlet', 'mexican_hat', or 'paul')
        omega0 : float
            Nondimensional frequency parameter for the Morlet wavelet
        m : int
            Order parameter for the Paul wavelet
        """
        self.data = data
        self.dt = dt
        self.wavelet = wavelet
        self.omega0 = omega0
        self.m = m
        
        # Compute basic statistics
        self.N = len(data)
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # Compute returns if provided with price data
        self.returns = None
        if self.N > 1:
            self.returns = np.diff(np.log(data)) if np.all(data > 0) else np.diff(data)
        
        # Initialize wavelet transform storage
        self.scales = None
        self.W = None
        self.power = None
        self.significance = None
        self.coi = None
        self.coi_mask = None
    
    def compute_autocorrelation(self, lag=1):
        """
        Compute lag-N autocorrelation coefficient.
        
        Parameters:
        -----------
        lag : int
            Lag for autocorrelation calculation
            
        Returns:
        --------
        autocorr : float
            Autocorrelation coefficient
        """
        # Ensure we have computed returns
        if self.returns is None:
            if np.all(self.data > 0):
                self.returns = np.diff(np.log(self.data))
            else:
                self.returns = np.diff(self.data)
        
        # Compute autocorrelation
        n = len(self.returns)
        y1 = self.returns[0:(n-lag)]
        y2 = self.returns[lag:n]
        
        # Return correlation coefficient
        return np.corrcoef(y1, y2)[0, 1]
    
    def compute_optimal_scales(self, min_period, max_period, num_scales):
        """
        Compute optimal scales for the wavelet transform based on desired periods.
        
        Parameters:
        -----------
        min_period : float
            Minimum period of interest (in same units as dt)
        max_period : float
            Maximum period of interest (in same units as dt)
        num_scales : int
            Number of scales to compute
            
        Returns:
        --------
        scales : array_like
            Optimal scales for the wavelet transform
        """
        # Calculate scale spacing in octaves
        dj = 1.0 / num_scales * np.log2(max_period / min_period)
        
        # Compute scales
        scales = np.zeros(num_scales)
        for j in range(num_scales):
            scales[j] = scale_from_period(min_period * 2**(j * dj), 
                                         wavelet=self.wavelet, 
                                         omega0=self.omega0, 
                                         m=self.m)
        
        self.scales = scales
        self.dj = dj
        
        return scales
    
    def perform_cwt_analysis(self, min_period=2, max_period=None, num_scales=12):
        """
        Perform continuous wavelet transform analysis on the financial data.
        
        Parameters:
        -----------
        min_period : float
            Minimum period of interest (in same units as dt)
        max_period : float
            Maximum period of interest (in same units as dt)
        num_scales : int
            Number of scales to compute
            
        Returns:
        --------
        power : array_like
            Wavelet power spectrum
        """
        # Default max period is 1/4 of the data length
        if max_period is None:
            max_period = self.N * self.dt / 4
        
        # Compute scales
        scales = self.compute_optimal_scales(min_period, max_period, num_scales)
        
        # Compute CWT
        self.W = cwt(self.data, self.dt, scales, wavelet=self.wavelet, 
                     omega0=self.omega0, m=self.m)
        
        # Compute power spectrum
        self.power = cwt_power(self.W)
        
        # Compute cone of influence
        self.coi = cone_of_influence(self.N, self.dt, scales, wavelet=self.wavelet, 
                                    omega0=self.omega0, m=self.m)
        
        # Create COI mask
        self.coi_mask = compute_coi_mask(self.power, scales, self.coi)
        
        return self.power
    
    def compute_significance(self, alpha=0.05, lag1=None):
        """
        Compute significance levels for the wavelet power spectrum.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default=0.05)
        lag1 : float, optional
            Lag-1 autocorrelation coefficient (if None, will be computed)
            
        Returns:
        --------
        significance : array_like
            Significance levels for the power spectrum
        """
        if lag1 is None:
            lag1 = self.compute_autocorrelation(lag=1)
        
        self.significance = cwt_significance(self.power, self.scales, self.dt, 
                                            alpha=alpha, lag1=lag1)
        
        return self.significance
    
    def detect_events(self, threshold_quantile=0.95, min_duration=1):
        """
        Detect significant events in the wavelet power spectrum.
        
        Parameters:
        -----------
        threshold_quantile : float
            Quantile threshold for significant power (default=0.95)
        min_duration : int
            Minimum duration of an event in time steps
            
        Returns:
        --------
        events : list of dict
            List of detected events with their properties
        """
        if self.power is None:
            raise ValueError("Wavelet analysis must be performed first")
        
        # Apply mask for cone of influence
        masked_power = self.power.copy()
        masked_power[~self.coi_mask] = 0
        
        # Calculate threshold
        threshold = np.quantile(masked_power[masked_power > 0], threshold_quantile)
        
        # Find regions above threshold
        events = []
        for i, scale in enumerate(self.scales):
            # Find consecutive points above threshold
            above_threshold = masked_power[i, :] > threshold
            
            # Find runs of True values
            runs = np.where(np.diff(np.concatenate(([False], above_threshold, [False]))))[0].reshape(-1, 2)
            
            # Filter by minimum duration
            runs = runs[runs[:, 1] - runs[:, 0] >= min_duration]
            
            # Extract event properties
            for start, end in runs:
                period = fourier_period(scale, wavelet=self.wavelet, 
                                      omega0=self.omega0, m=self.m)
                
                events.append({
                    'start': start,
                    'end': end - 1,  # Adjust for diff indexing
                    'duration': end - start,
                    'scale_index': i,
                    'scale': scale,
                    'period': period,
                    'max_power': np.max(masked_power[i, start:end]),
                    'avg_power': np.mean(masked_power[i, start:end])
                })
        
        # Sort events by max power
        events.sort(key=lambda x: x['max_power'], reverse=True)
        
        return events
    
    def volatility_analysis(self, window_size=None):
        """
        Perform wavelet-based volatility analysis.
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the rolling window for traditional volatility
            
        Returns:
        --------
        wavelet_volatility : array_like
            Wavelet-based volatility estimate
        rolling_volatility : array_like
            Traditional rolling window volatility (if window_size provided)
        """
        # Ensure we have computed returns
        if self.returns is None:
            if np.all(self.data > 0):
                self.returns = np.diff(np.log(self.data))
            else:
                self.returns = np.diff(self.data)
        
        # Compute traditional volatility if window size is provided
        rolling_volatility = None
        if window_size is not None:
            rolling_volatility = np.zeros(len(self.returns))
            rolling_volatility[:] = np.nan
            
            for i in range(window_size, len(self.returns) + 1):
                rolling_volatility[i-1] = np.std(self.returns[i-window_size:i])
        
        # Compute wavelet-based volatility
        # We focus on high-frequency scales for volatility
        high_freq_indices = np.where(self.scales < np.median(self.scales))[0]
        
        if len(high_freq_indices) > 0:
            # Use scale-averaged power for high-frequency scales
            wavelet_volatility = np.zeros(self.N)
            
            # For returns we need to adjust since we have one fewer point
            if len(self.returns) < self.N:
                power_for_returns = self.power[:, 1:]
                scale_avg = scale_averaged_power(
                    power_for_returns[high_freq_indices, :], 
                    self.scales[high_freq_indices], 
                    self.dj
                )
                wavelet_volatility[1:] = np.sqrt(scale_avg)
            else:
                scale_avg = scale_averaged_power(
                    self.power[high_freq_indices, :], 
                    self.scales[high_freq_indices], 
                    self.dj
                )
                wavelet_volatility = np.sqrt(scale_avg)
        else:
            # Fall back to full spectrum if no high-frequency scales
            scale_avg = scale_averaged_power(self.power, self.scales, self.dj)
            wavelet_volatility = np.sqrt(scale_avg)
        
        return wavelet_volatility, rolling_volatility
    
    def jump_detection(self, threshold=3.0):
        """
        Detect jumps in the financial time series using wavelet analysis.
        
        Parameters:
        -----------
        threshold : float
            Number of standard deviations for jump detection
            
        Returns:
        --------
        jumps : list of dict
            List of detected jumps with their properties
        """
        # Ensure we have computed returns
        if self.returns is None:
            if np.all(self.data > 0):
                self.returns = np.diff(np.log(self.data))
            else:
                self.returns = np.diff(self.data)
        
        # Get the power at the smallest scale (highest frequency)
        if self.power is None or self.scales is None:
            raise ValueError("Wavelet analysis must be performed first")
        
        # Find the smallest scale
        min_scale_idx = np.argmin(self.scales)
        smallest_scale_power = self.power[min_scale_idx, 1:]  # Adjust for returns length
        
        # Compute threshold
        power_mean = np.mean(smallest_scale_power)
        power_std = np.std(smallest_scale_power)
        power_threshold = power_mean + threshold * power_std
        
        # Detect jumps
        jump_indices = np.where(smallest_scale_power > power_threshold)[0]
        
        # Group consecutive jumps
        if len(jump_indices) == 0:
            return []
        
        jumps = []
        current_jump = [jump_indices[0]]
        
        for i in range(1, len(jump_indices)):
            if jump_indices[i] - jump_indices[i-1] <= 1:
                current_jump.append(jump_indices[i])
            else:
                # Process current jump
                avg_size = np.mean(np.abs(self.returns[current_jump]))
                jumps.append({
                    'indices': current_jump,
                    'start': current_jump[0],
                    'end': current_jump[-1],
                    'size': avg_size,
                    'power': np.mean(smallest_scale_power[current_jump]),
                    'returns': self.returns[current_jump]
                })
                current_jump = [jump_indices[i]]
        
        # Add the last jump
        avg_size = np.mean(np.abs(self.returns[current_jump]))
        jumps.append({
            'indices': current_jump,
            'start': current_jump[0],
            'end': current_jump[-1],
            'size': avg_size,
            'power': np.mean(smallest_scale_power[current_jump]),
            'returns': self.returns[current_jump]
        })
        
        # Sort jumps by power
        jumps.sort(key=lambda x: x['power'], reverse=True)
        
        return jumps
    
    def volatility_clustering_analysis(self, window_sizes=[5, 10, 20]):
        """
        Analyze volatility clustering using wavelet-based methods.
        
        Parameters:
        -----------
        window_sizes : list of int
            Sizes of windows for autocorrelation analysis
            
        Returns:
        --------
        results : dict
            Results of volatility clustering analysis
        """
        # Ensure we have computed returns
        if self.returns is None:
            if np.all(self.data > 0):
                self.returns = np.diff(np.log(self.data))
            else:
                self.returns = np.diff(self.data)
        
        # Compute squared returns (proxy for volatility)
        squared_returns = self.returns**2
        
        # Get wavelet-based volatility
        wavelet_vol, _ = self.volatility_analysis()
        
        # Calculate autocorrelations at different lags
        max_lag = max(window_sizes)
        acf_squared = np.zeros(max_lag)
        acf_wavelet = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            # Autocorrelation of squared returns
            acf_squared[lag-1] = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
            
            # Autocorrelation of wavelet-based volatility (skip first point due to returns calculation)
            vol = wavelet_vol[1:]
            if len(vol) > lag:
                acf_wavelet[lag-1] = np.corrcoef(vol[:-lag], vol[lag:])[0, 1]
        
        # Check for statistical significance
        n = len(self.returns)
        # Critical value at 95% confidence (1.96/sqrt(n))
        critical_value = 1.96 / np.sqrt(n)
        
        # Create results dictionary
        results = {
            'squared_returns_acf': acf_squared,
            'wavelet_volatility_acf': acf_wavelet,
            'critical_value': critical_value,
            'window_sizes': window_sizes,
            'significant_sq_returns': acf_squared > critical_value,
            'significant_wavelet': acf_wavelet > critical_value
        }
        
        return results
        
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