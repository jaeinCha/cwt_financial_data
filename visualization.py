"""
Visualization tools for wavelet analysis of financial time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from wavelet_base import fourier_period

def plot_time_series(time, data, title="Time Series", xlabel="Time", ylabel="Value", 
                     figsize=(12, 4), returns=False):
    """
    Plot a financial time series.
    
    Parameters:
    -----------
    time : array_like
        Time values (can be datetime)
    data : array_like
        Time series data
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    returns : bool
        Whether data is returns (True) or prices (False)
    """
    plt.figure(figsize=figsize)
    
    if returns:
        plt.plot(time, data, 'b-')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    else:
        plt.plot(time, data, 'b-')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    return plt.gca()

def plot_cwt_scalogram(time, scales, power, coi=None, period_range=None, 
                       title="Wavelet Power Spectrum", figsize=(12, 8),
                       cmap='jet', wavelet='morlet', omega0=6.0, m=4, 
                       significance=None, alpha=0.5, log_scale=True):
    """
    Plot the CWT scalogram (power spectrum).
    
    Parameters:
    -----------
    time : array_like
        Time values
    scales : array_like
        Wavelet scales
    power : array_like
        Wavelet power spectrum
    coi : array_like, optional
        Cone of influence
    period_range : tuple, optional
        Range of periods to display (min, max)
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the scalogram
    wavelet : str
        Wavelet type
    omega0, m : float, int
        Wavelet parameters
    significance : array_like, optional
        Significance levels
    alpha : float
        Transparency for significance contours
    log_scale : bool
        Whether to use logarithmic color scale
    """
    # Convert scales to periods
    periods = np.array([fourier_period(s, wavelet=wavelet, omega0=omega0, m=m) for s in scales])
    
    # Filter periods if range is provided
    if period_range is not None:
        min_period, max_period = period_range
        idx = (periods >= min_period) & (periods <= max_period)
        periods = periods[idx]
        power_plot = power[idx, :]
        if significance is not None:
            significance_plot = significance[idx, :]
    else:
        power_plot = power
        significance_plot = significance
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot power spectrum
    if log_scale:
        # Add small constant to avoid log(0)
        mesh = plt.pcolormesh(time, periods, power_plot, cmap=cmap, norm=LogNorm(vmin=np.max(power_plot)/100, vmax=np.max(power_plot)))
    else:
        mesh = plt.pcolormesh(time, periods, power_plot, cmap=cmap)
    
    # Plot significance contours
    if significance_plot is not None:
        plt.contour(time, periods, power_plot / significance_plot, levels=[1], colors='k', linewidths=2, alpha=alpha)
    
    # Plot cone of influence
    if coi is not None:
        # Convert COI to periods
        coi_periods = np.array([fourier_period(c, wavelet=wavelet, omega0=omega0, m=m) for c in coi], dtype=float)
        
        # If time is datetime64, convert to matplotlib dates for compatibility
        if np.issubdtype(np.array(time).dtype, np.datetime64):
            import matplotlib.dates as mdates
            time_numeric = mdates.date2num(time)
            plt.plot(time, coi_periods, 'k--')
            max_period_value = float(np.max(periods))
            ones_array = np.ones_like(time_numeric, dtype=float)
            plt.fill_between(time, coi_periods, max_period_value * ones_array, color='white', alpha=0.5)
        else:
            # Regular numeric time
            time_float = np.array(time, dtype=float)
            plt.plot(time_float, coi_periods, 'k--')
            max_period_value = float(np.max(periods))
            ones_array = np.ones_like(time_float, dtype=float)
            plt.fill_between(time_float, coi_periods, max_period_value * ones_array, color='white', alpha=0.5)
    
    # Set labels and title
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Period')
    plt.yscale('log')  # Log scale for periods
    plt.colorbar(mesh, label='Power')
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    return plt.gca()

def plot_wavelet_volatility(time, wavelet_volatility, rolling_volatility=None, window_size=None,
                           title="Volatility Analysis", figsize=(12, 4)):
    """
    Plot wavelet-based volatility compared to traditional volatility.
    
    Parameters:
    -----------
    time : array_like
        Time values
    wavelet_volatility : array_like
        Wavelet-based volatility
    rolling_volatility : array_like, optional
        Traditional rolling window volatility
    window_size : int, optional
        Size of rolling window (for label)
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Ensure arrays have matching lengths
    if len(time) != len(wavelet_volatility):
        # Adjust time array if necessary
        time = time[:len(wavelet_volatility)] if len(time) > len(wavelet_volatility) else time
        wavelet_volatility = wavelet_volatility[:len(time)] if len(wavelet_volatility) > len(time) else wavelet_volatility
    
    # Plot wavelet-based volatility
    plt.plot(time, wavelet_volatility, 'b-', label='Wavelet Volatility')
    
    # Plot rolling volatility if provided
    if rolling_volatility is not None:
        # For shorter rolling_volatility arrays (due to window), ensure proper time alignment
        if len(rolling_volatility) < len(time):
            # If window_size is provided, align at the end
            if window_size and window_size < len(time):
                roll_time = time[window_size-1:window_size-1+len(rolling_volatility)]
            else:
                # If window size not provided or too large, align at the end
                roll_time = time[-len(rolling_volatility):]
        else:
            # Truncate rolling_volatility if needed
            rolling_volatility = rolling_volatility[:len(time)]
            roll_time = time
        
        # Ensure roll_time and rolling_volatility have the same length
        min_len = min(len(roll_time), len(rolling_volatility))
        roll_time = roll_time[:min_len]
        rolling_volatility = rolling_volatility[:min_len]
        
        # Remove NaN values
        valid_indices = ~np.isnan(rolling_volatility)
        
        # # Only plot if there are valid values
        # if np.any(valid_indices):
        #     window_label = f"{window_size}-period" if window_size else "Rolling"
        #     plt.plot(roll_time[valid_indices], rolling_volatility[valid_indices], 
        #             'r--', label=f'{window_label} Volatility')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    return plt.gca()

def plot_scale_averaged_power(time, scale_avg_power, time_series=None, period_band=None,
                              title="Scale-Averaged Power", figsize=(12, 8)):
    """
    Plot scale-averaged wavelet power in a range of periods.
    
    Parameters:
    -----------
    time : array_like
        Time values
    scale_avg_power : array_like
        Scale-averaged power
    time_series : array_like, optional
        Original time series for comparison
    period_band : tuple, optional
        Period band used for averaging (for title)
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot original time series if provided
    if time_series is not None:
        ax1 = fig.add_subplot(211)
        ax1.plot(time, time_series, 'b-')
        ax1.set_ylabel('Price')
        ax1.set_title('Time Series')
        ax1.grid(True, alpha=0.3)
        
        # Format date ticks if datetime
        if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax2 = fig.add_subplot(212, sharex=ax1)
    else:
        ax2 = fig.add_subplot(111)
    
    # Plot scale-averaged power
    ax2.plot(time, scale_avg_power, 'r-')
    
    # Add period band to title if provided
    if period_band is not None:
        title = f"{title} ({period_band[0]}-{period_band[1]} period band)"
    
    ax2.set_title(title)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Power')
    ax2.grid(True, alpha=0.3)
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return ax2

def plot_volatility_clustering(lags, acf_squared, acf_wavelet, critical_value,
                              title="Volatility Clustering Analysis", figsize=(10, 6)):
    """
    Plot autocorrelation functions to visualize volatility clustering.
    
    Parameters:
    -----------
    lags : array_like
        Lag values
    acf_squared : array_like
        Autocorrelation of squared returns
    acf_wavelet : array_like
        Autocorrelation of wavelet-based volatility
    critical_value : float
        Critical value for statistical significance
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot autocorrelations
    plt.plot(lags, acf_squared, 'b-o', label='Squared Returns ACF')
    plt.plot(lags, acf_wavelet, 'r-s', label='Wavelet Volatility ACF')
    
    # Plot significance levels
    plt.axhline(y=critical_value, color='k', linestyle='--', alpha=0.7, label='95% Confidence')
    plt.axhline(y=-critical_value, color='k', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(lags))
    plt.xticks(lags)
    
    plt.tight_layout()
    
    return plt.gca()

def plot_jump_detection(time, data, jumps, title="Jump Detection", figsize=(12, 6)):
    """
    Plot detected jumps in financial time series.
    
    Parameters:
    -----------
    time : array_like
        Time values
    data : array_like
        Time series data
    jumps : list of dict
        List of detected jumps
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot the time series
    plt.plot(time, data, 'b-', alpha=0.7)
    
    # Highlight jumps
    for jump in jumps:
        # Adjust for returns calculation offset if needed
        idx = jump['indices']
        if len(data) > len(time):
            # If data is returns, adjust indices
            idx = [i+1 for i in idx]
        
        # Plot jump points
        plt.plot(time[idx], data[idx], 'ro', markersize=8)
        
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    return plt.gca()

def create_wavelet_dashboard(analyzer, time, figsize=(15, 10)):
    """
    Create a comprehensive dashboard of wavelet analysis results.
    
    Parameters:
    -----------
    analyzer : FinancialWaveletAnalyzer
        Analyzer object with completed analysis
    time : array_like
        Time values
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Define grid
    gs = fig.add_gridspec(3, 2)
    
    # Plot original time series
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, analyzer.data, 'b-')
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('')
    ax1.grid(True, alpha=0.3)
    
    # Plot returns
    if analyzer.returns is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        returns_time = time[1:] if len(analyzer.returns) < len(time) else time
        ax2.plot(returns_time, analyzer.returns, 'g-')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('Returns')
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3)
    
    # Plot wavelet scalogram
    ax3 = fig.add_subplot(gs[1, :])
    
    # Convert scales to periods
    periods = np.array([fourier_period(s, wavelet=analyzer.wavelet, 
                                      omega0=analyzer.omega0, m=analyzer.m) 
                        for s in analyzer.scales], dtype=float)
    
    # Convert to float64 to avoid type issues
    power_plot = analyzer.power.astype(np.float64)
    
    # Plot scalogram
    mesh = ax3.pcolormesh(time, periods, power_plot, 
                         cmap='jet', 
                         norm=LogNorm(vmin=np.max(power_plot)/100, 
                                    vmax=np.max(power_plot)))
    
    # Plot significance contours if available
    if analyzer.significance is not None:
        # Convert to same data type
        significance_plot = analyzer.significance.astype(np.float64)
        ax3.contour(time, periods, power_plot / significance_plot, 
                   levels=[1], colors='k', linewidths=2, alpha=0.7)
    
    # Plot cone of influence
    if analyzer.coi is not None:
        # Convert COI to periods
        coi_periods = np.array([fourier_period(c, wavelet=analyzer.wavelet, 
                                             omega0=analyzer.omega0, m=analyzer.m) 
                               for c in analyzer.coi], dtype=float)
        
        ax3.plot(time, coi_periods, 'k--')
        
        # Shade area outside COI
        # If time is datetime64, convert for compatibility
        if np.issubdtype(np.array(time).dtype, np.datetime64):
            import matplotlib.dates as mdates
            max_period_value = float(np.max(periods))
            ones_array = np.ones_like(coi_periods, dtype=float)
            ax3.fill_between(time, coi_periods, max_period_value, color='white', alpha=0.5)
        else:
            # Regular numeric time
            time_float = np.array(time, dtype=float)
            max_period_value = float(np.max(periods))
            ones_array = np.ones_like(time_float, dtype=float)
            ax3.fill_between(time_float, coi_periods, max_period_value * ones_array, color='white', alpha=0.5)
    
    ax3.set_title('Wavelet Power Spectrum')
    ax3.set_ylabel('Period')
    ax3.set_yscale('log')
    plt.colorbar(mesh, ax=ax3, label='Power')
    
    # Plot wavelet-based volatility
    # Define window size for rolling volatility
    window_size = 20
    wavelet_vol, rolling_vol = analyzer.volatility_analysis(window_size=window_size)
    
    ax4 = fig.add_subplot(gs[2, 0])
    vol_time = time[1:] if len(wavelet_vol) < len(time) else time
    ax4.plot(vol_time[:len(wavelet_vol)], wavelet_vol[:len(vol_time)], 'b-', label='Wavelet Volatility')
    
    if rolling_vol is not None:
        # For shorter rolling_volatility arrays (due to window), ensure proper time alignment
        if len(rolling_vol) < len(time):
            # If window_size is provided, align at the end
            if window_size and window_size < len(time):
                roll_time = time[window_size-1:min(window_size-1+len(rolling_vol), len(time))]
            else:
                # If window size not provided or too large, align at the end
                roll_time = time[-min(len(rolling_vol), len(time)):]
        else:
            # Truncate rolling_volatility if needed
            rolling_vol = rolling_vol[:len(time)]
            roll_time = time
        
        # Ensure roll_time and rolling_volatility have the same length
        min_len = min(len(roll_time), len(rolling_vol))
        roll_time = roll_time[:min_len]
        rolling_vol = rolling_vol[:min_len]
        
        # Remove NaN values
        valid_indices = ~np.isnan(rolling_vol)
        
        # Only plot if there are valid values and arrays match in dimension
        # if np.any(valid_indices) and len(roll_time) == len(valid_indices):
        #     ax4.plot(roll_time[valid_indices], rolling_vol[valid_indices], 
        #             'r--', label='20-period Rolling Volatility')
    
    ax4.set_title('Volatility Analysis')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot volatility clustering (ACF)
    clustering_results = analyzer.volatility_clustering_analysis()
    lags = np.arange(1, len(clustering_results['squared_returns_acf']) + 1)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(lags, clustering_results['squared_returns_acf'], 'b-o', 
            label='Squared Returns ACF', markersize=4)
    ax5.plot(lags, clustering_results['wavelet_volatility_acf'], 'r-s', 
            label='Wavelet Volatility ACF', markersize=4)
    ax5.axhline(y=clustering_results['critical_value'], color='k', 
                linestyle='--', alpha=0.7, label='95% Confidence')
    ax5.axhline(y=-clustering_results['critical_value'], color='k', 
                linestyle='--', alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_title('Volatility Clustering')
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('Autocorrelation')
    ax5.legend(fontsize='small')
    ax5.grid(True, alpha=0.3)
    
    # Format date ticks if datetime
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return fig