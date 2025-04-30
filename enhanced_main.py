"""
Enhanced main script for analyzing S&P 500 data using the Continuous Wavelet Transform.
Includes improvements for edge effect reduction and enhanced jump detection.
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
from edge_effects_handling import (extend_time_series, process_extended_results, 
                                 taper_time_series, boundary_aware_coi)
from enhanced_jump_detection import (adaptive_threshold_jump_detection, 
                                   multi_scale_jump_detection,
                                   volatility_aware_jump_detection,
                                   combined_jump_detection)

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

def analyze_sp500_with_reduced_edge_effects(time, data, output_dir="results"):
    """
    Perform wavelet analysis on S&P 500 data with reduced edge effects.
    
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
    
    # 1. Apply tapering to reduce edge discontinuities
    tapered_data = taper_time_series(data, taper_type='cosine')
    
    # 2. Extend the time series to reduce edge effects
    extended_data, extension_indices = extend_time_series(tapered_data, 
                                                         extension_type='reflection')
    
    # Create extended time array (for visualization only)
    orig_n = len(time)
    ext_n = len(extended_data)
    ext_left = extension_indices[0]
    
    # Assume uniform time steps for the extension
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        # For datetime objects, calculate the average time delta
        if isinstance(time[0], np.datetime64):
            time_diff = np.diff(time).astype('timedelta64[D]').astype(float).mean()
            extended_time = np.array([time[0] - np.timedelta64(int((ext_left-i)*time_diff), 'D') 
                                     for i in range(ext_left)] +
                                    list(time) +
                                    [time[-1] + np.timedelta64(int((i+1)*time_diff), 'D') 
                                     for i in range(ext_n - ext_left - orig_n)])
        else:
            # For other datetime objects
            import datetime
            time_diff = sum((time[i+1] - time[i]).days for i in range(len(time)-1)) / (len(time)-1)
            extended_time = [time[0] - datetime.timedelta(days=int((ext_left-i)*time_diff)) 
                           for i in range(ext_left)] + list(time) + \
                           [time[-1] + datetime.timedelta(days=int((i+1)*time_diff)) 
                            for i in range(ext_n - ext_left - orig_n)]
    else:
        # For numeric time values
        time_diff = np.diff(time).mean()
        extended_time = np.concatenate([
            time[0] - np.arange(ext_left, 0, -1) * time_diff,
            time,
            time[-1] + np.arange(1, ext_n - ext_left - orig_n + 1) * time_diff
        ])
    
    # Initialize the analyzer with the extended data
    analyzer = FinancialWaveletAnalyzer(extended_data, dt=1.0, wavelet='morlet', omega0=6.0)
    
    # 3. Compute wavelet transform
    # Define a range of periods to analyze
    min_period = 5  # 5 days (week)
    max_period = min(252, len(data) // 4)  # 1 trading year or 1/4 of data length
    num_scales = 32  # Number of scales to compute
    
    # Perform CWT analysis
    power = analyzer.perform_cwt_analysis(min_period=min_period, 
                                        max_period=max_period, 
                                        num_scales=num_scales)
    
    # Compute more conservative COI
    extended_coi = boundary_aware_coi(len(extended_data), 1.0, analyzer.scales)
    
    # Create COI mask for extended data
    from cwt_transform import compute_coi_mask
    analyzer.coi_mask = compute_coi_mask(analyzer.power, analyzer.scales, extended_coi)
    
    # Compute significance levels
    significance = analyzer.compute_significance(alpha=0.05)
    
    # 4. Process the results to extract the original portion
    analyzer.power = process_extended_results(analyzer.power, extension_indices)
    analyzer.W = process_extended_results(analyzer.W, extension_indices)
    analyzer.significance = process_extended_results(analyzer.significance, extension_indices)
    
    # Process the COI and COI mask for the original data
    if hasattr(analyzer, 'coi'):
        analyzer.coi = extended_coi[extension_indices[0]:extension_indices[1]]
    if hasattr(analyzer, 'coi_mask'):
        # Make sure to process the mask to match the new power dimensions
        analyzer.coi_mask = process_extended_results(analyzer.coi_mask, extension_indices)
    
    # Reset analyzer data to the original (unextended) data
    analyzer.data = data
    analyzer.N = len(data)
    
    # Recalculate returns on the original data
    if np.all(data > 0):
        analyzer.returns = np.diff(np.log(data))
    else:
        analyzer.returns = np.diff(data)
    
    # 5. Enhanced jump detection with multiple methods
    # Use all three methods, a combined approach
    enhanced_jumps = combined_jump_detection(analyzer, 
                                           methods=['adaptive', 'multi_scale', 'volatility'])
    
    # Normal jumps for comparison
    regular_jumps = analyzer.jump_detection(threshold=3.0)
    
    # 6. Start visualization and analysis
    # Plot the original time series
    fig1 = plt.figure(figsize=(12, 10))
    
    # Original price series
    ax1 = fig1.add_subplot(211)
    ax1 = plot_time_series(time, data, title="S&P 500 Index", 
                         xlabel="", ylabel="Price", figsize=None, 
                         returns=False)
    
    # Calculate and plot returns
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax2 = plot_time_series(time[1:], analyzer.returns, title="S&P 500 Returns", 
                         xlabel="Time", ylabel="Log Returns", figsize=None,
                         returns=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sp500_time_series.png"), dpi=300)
    
    # 7. Plot wavelet power spectrum with reduced edge effects
    plt.figure(figsize=(12, 8))
    ax = plot_cwt_scalogram(time, analyzer.scales, analyzer.power, coi=analyzer.coi,
                          title="S&P 500 Wavelet Power Spectrum (Reduced Edge Effects)", 
                          significance=analyzer.significance)
    plt.savefig(os.path.join(output_dir, "sp500_wavelet_scalogram_reduced_edges.png"), dpi=300)
    
    # 8. Plot jump detection comparison
    plt.figure(figsize=(12, 6))
    # plt.plot(time, data, 'b-', alpha=0.7)
    
    # # Plot regular jumps (original method)
    # regular_times = []
    # for jump in regular_jumps:
    #     idx = jump['indices']
    #     regular_times.extend([time[i+1] for i in idx])  # +1 for returns offset
    #     plt.plot([time[i+1] for i in idx], [data[i+1] for i in idx], 'go', markersize=8, alpha=0.6, label='Original Detection')
    
    # Plot enhanced jumps
    enhanced_times = []
    for jump in enhanced_jumps:
        idx = jump['indices']
        enhanced_times.extend([time[i+1] for i in idx])  # +1 for returns offset
        plt.plot([time[i+1] for i in idx], [data[i+1] for i in idx], 'ro', markersize=8, alpha=0.5, label='Detected Jump')
    
    plt.plot(time, data, 'b-', alpha=0.7)
    
    # Create custom legend without duplicates
    from matplotlib.lines import Line2D
    legend_elements = [
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Original Detection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Detected Jump')
    ]
    plt.legend(handles=legend_elements)
    
    plt.title("S&P 500 Jump Detection")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "sp500_jump_detection_comparison.png"), dpi=300)
    
    # 9. Analyze volatility using wavelets
    window_size = 20  # For comparison with traditional volatility
    wavelet_vol, rolling_vol = analyzer.volatility_analysis(window_size=window_size)
    
    plt.figure(figsize=(12, 6))
    vol_time = time[1:] if len(wavelet_vol) < len(time) else time
    ax = plot_wavelet_volatility(vol_time, wavelet_vol, 
                                rolling_volatility=rolling_vol,
                                window_size=window_size,
                                title="S&P 500 Volatility Analysis")
    plt.savefig(os.path.join(output_dir, "sp500_volatility.png"), dpi=300)
    
    # 10. Analyze specific period bands
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
    
    # 11. Study the enhanced jumps in detail
    print("\nEnhanced jump detection results:")
    for i, jump in enumerate(enhanced_jumps[:15]):  # Show top 15 jumps
        jump_time = time[jump['start']+1]  # +1 for returns offset
        methods = jump.get('methods', [jump.get('method', 'unknown')])
        if isinstance(methods, list):
            method_str = ', '.join(set(methods))
        else:
            method_str = methods
            
        print(f"Jump {i+1}: Time: {jump_time}, Size: {jump['size']:.4f}, Method: {method_str}")
    
    print(f"\nTotal jumps detected: {len(enhanced_jumps)}")
    print(f"Results saved to {output_dir} directory")
    
    # 12. Create comprehensive dashboard with enhanced analysis
    fig = create_wavelet_dashboard(analyzer, time)
    plt.savefig(os.path.join(output_dir, "sp500_wavelet_dashboard_enhanced.png"), dpi=300)
    
    plt.close('all')
    return analyzer, enhanced_jumps

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
    
    # Perform enhanced analysis
    print("Performing wavelet analysis with reduced edge effects and enhanced jump detection...")
    analyzer, jumps = analyze_sp500_with_reduced_edge_effects(time, data)
    
    print("\nAnalysis complete!")
    
    # Print out COVID-specific events
    if isinstance(time[0], np.datetime64) or hasattr(time[0], 'year'):
        try:
            # Find COVID period (approximate date range for COVID crash)
            covid_start = np.datetime64('2020-02-15') if isinstance(time[0], np.datetime64) else datetime(2020, 2, 15)
            covid_end = np.datetime64('2020-04-15') if isinstance(time[0], np.datetime64) else datetime(2020, 4, 15)
            
            covid_jumps = []
            for jump in jumps:
                jump_time = time[jump['start']+1]  # +1 for returns offset
                if covid_start <= jump_time <= covid_end:
                    covid_jumps.append(jump)
            
            if covid_jumps:
                print("\nCOVID-19 related market events detected:")
                for i, jump in enumerate(covid_jumps):
                    jump_time = time[jump['start']+1]
                    methods = jump.get('methods', [jump.get('method', 'unknown')])
                    if isinstance(methods, list):
                        method_str = ', '.join(set(methods))
                    else:
                        method_str = methods
                    print(f"Event {i+1}: Time: {jump_time}, Size: {jump['size']:.4f}, Method: {method_str}")
        except (ValueError, TypeError):
            print("Could not analyze COVID-specific events due to datetime comparison issues")
# analyzer.power = process_extended_results(analyzer.power, extension_indices)
# analyzer.W = process_extended_results(analyzer.W, extension_indices)
# analyzer.significance = process_extended_results(analyzer.significance, extension_indices)