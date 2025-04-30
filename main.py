"""
Main script for analyzing S&P 500 data using the Continuous Wavelet Transform.
Based on the methodologies in:
- Torrence and Compo (1998) "A Practical Guide to Wavelet Analysis"
- Eliasson (2018) "An Application of the Continuous Wavelet Transform to Financial Time Series"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import our modules
from analyze_financial import FinancialWaveletAnalyzer, scale_averaged_power
from visualization import (plot_time_series, plot_cwt_scalogram, plot_wavelet_volatility,
                         plot_scale_averaged_power, plot_volatility_clustering,
                         plot_jump_detection, create_wavelet_dashboard)

def load_sp500_data(file_path):
    """
    Load S&P 500 data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
        
    Returns:
    --------
    time : array_like
        Time values
    data : array_like
        S&P 500 price data
    """
    try:
        # Try to read Excel file
        df = pd.read_excel(file_path)
        
        # Check if the data has expected columns
        if df.shape[1] < 2:
            raise ValueError("Excel file does not have expected columns")
        
        # Rename columns if needed
        if 'Date' not in df.columns:
            df.columns = ['Date', 'Price']
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Extract time and data
        time = df['Date'].values
        data = df.iloc[:, 1].values
        
        return time, data
    
    except Exception as e:
        print(f"Error loading S&P 500 data: {e}")
        
        # Create synthetic data for testing if file cannot be loaded
        print("Creating synthetic data for testing...")
        n_points = 1000
        time = np.array([datetime(2015, 1, 1) + pd.Timedelta(days=i) for i in range(n_points)])
        
        # Generate synthetic price series with trend, cycle, and noise
        t = np.linspace(0, 4*np.pi, n_points)
        trend = 1000 + t * 50
        cycle1 = 50 * np.sin(t)
        cycle2 = 30 * np.sin(5*t)
        noise = np.random.normal(0, 20, n_points)
        jumps = np.zeros_like(t)
        jump_points = [100, 300, 700]
        for jp in jump_points:
            jumps[jp] = 100 * np.random.choice([-1, 1])
        
        data = trend + cycle1 + cycle2 + noise + jumps.cumsum()
        
        return time, data

def analyze_sp500(time, data, output_dir="results"):
    """
    Perform wavelet analysis on S&P 500 data.
    
    Parameters:
    -----------
    time : array_like
        Time values
    data : array_like
        S&P 500 price data
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = FinancialWaveletAnalyzer(data, dt=1.0, wavelet='morlet', omega0=6.0)
    
    # 1. Plot the original time series
    fig1 = plt.figure(figsize=(12, 10))
    
    # Original price series
    ax1 = fig1.add_subplot(211)
    ax1 = plot_time_series(time, data, title="S&P 500 Index", 
                         xlabel="", ylabel="Price", figsize=None, 
                         returns=False)
    
    # Calculate and plot returns
    if np.all(data > 0):
        returns = np.diff(np.log(data))
    else:
        returns = np.diff(data)
    
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax2 = plot_time_series(time[1:], returns, title="S&P 500 Returns", 
                         xlabel="Time", ylabel="Log Returns", figsize=None,
                         returns=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sp500_time_series.png"), dpi=300)
    
    # 2. Compute wavelet transform
    # Define a range of periods to analyze
    # For financial data, we might be interested in periods from 5 days to 1 year
    min_period = 5  # 5 days (week)
    max_period = min(252, len(data) // 4)  # 1 trading year or 1/4 of data length
    num_scales = 32  # Number of scales to compute
    
    # Perform CWT analysis
    power = analyzer.perform_cwt_analysis(min_period=min_period, 
                                        max_period=max_period, 
                                        num_scales=num_scales)
    
    # Compute significance levels
    significance = analyzer.compute_significance(alpha=0.05)
    
    # 3. Plot wavelet power spectrum
    plt.figure(figsize=(12, 8))
    ax = plot_cwt_scalogram(time, analyzer.scales, analyzer.power, coi=analyzer.coi,
                          title="S&P 500 Wavelet Power Spectrum", 
                          significance=analyzer.significance)
    plt.savefig(os.path.join(output_dir, "sp500_wavelet_scalogram.png"), dpi=300)
    
    # 4. Analyze volatility using wavelets
    window_size = 20  # For comparison with traditional volatility
    wavelet_vol, rolling_vol = analyzer.volatility_analysis(window_size=window_size)
    
    plt.figure(figsize=(12, 6))
    vol_time = time[1:] if len(wavelet_vol) < len(time) else time
    ax = plot_wavelet_volatility(vol_time, wavelet_vol, 
                                rolling_volatility=rolling_vol,
                                window_size=window_size,
                                title="S&P 500 Volatility Analysis")
    plt.savefig(os.path.join(output_dir, "sp500_volatility.png"), dpi=300)
    
    # 5. Detect jumps in the price series
    jumps = analyzer.jump_detection(threshold=3.0)
    
    plt.figure(figsize=(12, 6))
    ax = plot_jump_detection(time, data, jumps, 
                           title="S&P 500 Jump Detection")
    plt.savefig(os.path.join(output_dir, "sp500_jumps.png"), dpi=300)
    
    # 6. Analyze volatility clustering
    clustering_results = analyzer.volatility_clustering_analysis(window_sizes=[5, 10, 20])
    
    plt.figure(figsize=(10, 6))
    lags = np.arange(1, len(clustering_results['squared_returns_acf']) + 1)
    ax = plot_volatility_clustering(lags, 
                                   clustering_results['squared_returns_acf'],
                                   clustering_results['wavelet_volatility_acf'],
                                   clustering_results['critical_value'],
                                   title="S&P 500 Volatility Clustering")
    plt.savefig(os.path.join(output_dir, "sp500_volatility_clustering.png"), dpi=300)
    
    # 7. Create comprehensive dashboard
    fig = create_wavelet_dashboard(analyzer, time)
    plt.savefig(os.path.join(output_dir, "sp500_wavelet_dashboard.png"), dpi=300)
    
    # 8. Analyze specific period bands
    # Short-term dynamics (1-4 weeks)
    short_term_indices = np.where((analyzer.scales >= scale_from_period(5)) & 
                                 (analyzer.scales <= scale_from_period(20)))[0]
    
    if len(short_term_indices) > 0:
        short_term_power = analyzer.power[short_term_indices, :]
        short_term_avg = scale_averaged_power(short_term_power, 
                                            analyzer.scales[short_term_indices],
                                            analyzer.dj)
        
        plt.figure(figsize=(12, 6))
        ax = plot_scale_averaged_power(time, short_term_avg, time_series=data,
                                     period_band=(5, 20),
                                     title="Short-Term Market Dynamics")
        plt.savefig(os.path.join(output_dir, "sp500_short_term.png"), dpi=300)
    
    # Medium-term dynamics (1-6 months)
    medium_term_indices = np.where((analyzer.scales >= scale_from_period(20)) & 
                                  (analyzer.scales <= scale_from_period(120)))[0]
    
    if len(medium_term_indices) > 0:
        medium_term_power = analyzer.power[medium_term_indices, :]
        medium_term_avg = scale_averaged_power(medium_term_power,
                                             analyzer.scales[medium_term_indices],
                                             analyzer.dj)
        
        plt.figure(figsize=(12, 6))
        ax = plot_scale_averaged_power(time, medium_term_avg, time_series=data,
                                     period_band=(20, 120),
                                     title="Medium-Term Market Dynamics")
        plt.savefig(os.path.join(output_dir, "sp500_medium_term.png"), dpi=300)
    
    # 9. Detect significant events
    events = analyzer.detect_events(threshold_quantile=0.95, min_duration=3)
    
    # Print detected events
    print("\nDetected significant events:")
    for i, event in enumerate(events[:10]):  # Show top 10 events
        event_time = time[event['start']]
        period = event['period']
        print(f"Event {i+1}: Time: {event_time}, Period: {period:.1f} days, Duration: {event['duration']} days")
    
    print(f"\nTotal events detected: {len(events)}")
    print(f"Results saved to {output_dir} directory")
    
    plt.close('all')
    return analyzer

def scale_from_period(period, wavelet='morlet', omega0=6.0):
    """Helper function to convert period to scale."""
    if wavelet == 'morlet':
        return (period * (omega0 + np.sqrt(2 + omega0**2))) / (4 * np.pi)
    else:
        raise ValueError(f"Wavelet {wavelet} not implemented for period conversion")

if __name__ == "__main__":
    # Path to S&P 500 data file
    file_path = "PerformanceGraphExport.xls"
    
    # Load data
    print(f"Loading S&P 500 data from {file_path}...")
    time, data = load_sp500_data(file_path)
    print(f"Loaded {len(data)} data points")
    
    # Perform analysis
    print("Performing wavelet analysis...")
    analyzer = analyze_sp500(time, data)
    
    print("\nAnalysis complete!")