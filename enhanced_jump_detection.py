"""
Enhanced jump detection methods using wavelets with adaptive thresholds and multi-scale analysis.
"""

import numpy as np
from scipy import stats

def adaptive_threshold_jump_detection(analyzer, base_threshold=2.5, window_size=20, adaptivity=0.5):
    """
    Detect jumps using an adaptive threshold based on local volatility.
    
    Parameters:
    -----------
    analyzer : FinancialWaveletAnalyzer
        Analyzer object with completed wavelet transform
    base_threshold : float
        Base threshold value (lower than the original 3.0 for higher sensitivity)
    window_size : int
        Size of the rolling window for local volatility calculation
    adaptivity : float
        Weight for the local adjustment (0 = fixed threshold, 1 = fully adaptive)
        
    Returns:
    --------
    jumps : list of dict
        List of detected jumps with their properties
    """
    # Ensure wavelet analysis has been performed
    if analyzer.power is None:
        raise ValueError("Wavelet analysis must be performed first")
    
    # Ensure we have returns
    if analyzer.returns is None:
        if np.all(analyzer.data > 0):
            analyzer.returns = np.diff(np.log(analyzer.data))
        else:
            analyzer.returns = np.diff(analyzer.data)
    
    # Calculate rolling volatility
    rolling_std = np.zeros_like(analyzer.returns)
    returns_abs = np.abs(analyzer.returns)
    
    for i in range(len(rolling_std)):
        start_idx = max(0, i - window_size)
        rolling_std[i] = np.std(analyzer.returns[start_idx:i+1])
    
    # Get the smallest scale power (highest frequency)
    min_scale_idx = np.argmin(analyzer.scales)
    high_freq_power = analyzer.power[min_scale_idx, 1:]  # Adjust for returns length
    
    # Compute thresholds for each point
    global_std = np.std(high_freq_power)
    thresholds = np.zeros_like(high_freq_power)
    
    for i in range(len(thresholds)):
        # Combine global and local thresholds
        local_adjustment = rolling_std[i] / np.mean(rolling_std) if np.mean(rolling_std) > 0 else 1.0
        thresholds[i] = base_threshold * ((1 - adaptivity) + adaptivity * local_adjustment)
    
    # Detect jumps using adaptive thresholds
    jump_indices = np.where(high_freq_power > global_std * thresholds)[0]
    
    # Group consecutive jumps
    jumps = []
    if len(jump_indices) == 0:
        return jumps
    
    current_jump = [jump_indices[0]]
    
    for i in range(1, len(jump_indices)):
        if jump_indices[i] - jump_indices[i-1] <= 1:
            current_jump.append(jump_indices[i])
        else:
            # Process current jump
            avg_size = np.mean(np.abs(analyzer.returns[current_jump]))
            jumps.append({
                'indices': current_jump,
                'start': current_jump[0],
                'end': current_jump[-1],
                'size': avg_size,
                'power': np.mean(high_freq_power[current_jump]),
                'returns': analyzer.returns[current_jump],
                'threshold': np.mean(thresholds[current_jump]) * global_std
            })
            current_jump = [jump_indices[i]]
    
    # Add the last jump
    if current_jump:
        avg_size = np.mean(np.abs(analyzer.returns[current_jump]))
        jumps.append({
            'indices': current_jump,
            'start': current_jump[0],
            'end': current_jump[-1],
            'size': avg_size,
            'power': np.mean(high_freq_power[current_jump]),
            'returns': analyzer.returns[current_jump],
            'threshold': np.mean(thresholds[current_jump]) * global_std
        })
    
    # Sort jumps by power
    jumps.sort(key=lambda x: x['power'], reverse=True)
    
    return jumps

def multi_scale_jump_detection(analyzer, scales_to_use=3, base_threshold=2.5, min_duration=1):
    """
    Detect jumps by analyzing multiple scales, not just the smallest one.
    
    Parameters:
    -----------
    analyzer : FinancialWaveletAnalyzer
        Analyzer object with completed wavelet transform
    scales_to_use : int
        Number of smallest scales to use for jump detection
    base_threshold : float
        Base threshold value
    min_duration : int
        Minimum duration of an event in time steps
        
    Returns:
    --------
    jumps : list of dict
        List of detected jumps with their properties
    """
    # Ensure wavelet analysis has been performed
    if analyzer.power is None:
        raise ValueError("Wavelet analysis must be performed first")
    
    # Ensure we have returns
    if analyzer.returns is None:
        if np.all(analyzer.data > 0):
            analyzer.returns = np.diff(np.log(analyzer.data))
        else:
            analyzer.returns = np.diff(analyzer.data)
    
    # Get power at the smallest scales
    scale_indices = np.argsort(analyzer.scales)[:scales_to_use]
    scale_indices.sort()  # Keep them in order
    
    # Create a copy of the power array
    masked_power = analyzer.power.copy()
    
    # Apply COI mask to remove edge effects only if it exists and dimensions match
    if hasattr(analyzer, 'coi_mask') and analyzer.coi_mask is not None:
        # Check if dimensions match before applying the mask
        if analyzer.coi_mask.shape == masked_power.shape:
            masked_power[~analyzer.coi_mask] = 0
        else:
            print("Warning: COI mask dimensions don't match power array. Skipping COI masking.")
    
    # Create weighted multi-scale power
    multi_scale_power = np.zeros(analyzer.power.shape[1] - 1)  # For returns length
    
    for i, scale_idx in enumerate(scale_indices):
        # Weight smaller scales higher
        weight = 1.0 / (i + 1)
        scale_power = masked_power[scale_idx, 1:]  # Adjust for returns
        
        # Normalize by scale standard deviation
        scale_std = np.std(scale_power)
        if scale_std > 0:
            normalized_power = scale_power / scale_std
        else:
            normalized_power = scale_power
        
        multi_scale_power += weight * normalized_power
    
    # Calculate threshold
    power_std = np.std(multi_scale_power)
    power_mean = np.mean(multi_scale_power)
    threshold = power_mean + base_threshold * power_std
    
    # Detect jumps
    jump_indices = np.where(multi_scale_power > threshold)[0]
    
    # Group consecutive jumps
    jumps = []
    if len(jump_indices) == 0:
        return jumps
    
    current_jump = [jump_indices[0]]
    
    for i in range(1, len(jump_indices)):
        if jump_indices[i] - jump_indices[i-1] <= 1:
            current_jump.append(jump_indices[i])
        else:
            # Process current jump if it meets minimum duration
            if len(current_jump) >= min_duration:
                avg_size = np.mean(np.abs(analyzer.returns[current_jump]))
                jumps.append({
                    'indices': current_jump,
                    'start': current_jump[0],
                    'end': current_jump[-1],
                    'size': avg_size,
                    'power': np.mean(multi_scale_power[current_jump]),
                    'scales': [analyzer.scales[idx] for idx in scale_indices],
                    'returns': analyzer.returns[current_jump]
                })
            current_jump = [jump_indices[i]]
    
    # Add the last jump if it meets minimum duration
    if current_jump and len(current_jump) >= min_duration:
        avg_size = np.mean(np.abs(analyzer.returns[current_jump]))
        jumps.append({
            'indices': current_jump,
            'start': current_jump[0],
            'end': current_jump[-1],
            'size': avg_size,
            'power': np.mean(multi_scale_power[current_jump]),
            'scales': [analyzer.scales[idx] for idx in scale_indices],
            'returns': analyzer.returns[current_jump]
        })
    
    # Sort jumps by power
    jumps.sort(key=lambda x: x['power'], reverse=True)
    
    return jumps

def volatility_aware_jump_detection(analyzer, window_size=20, threshold_multiplier=3.0):
    """
    Detect jumps by comparing returns to local volatility.
    This is particularly effective for catching events like the COVID crash.
    
    Parameters:
    -----------
    analyzer : FinancialWaveletAnalyzer
        Analyzer object with completed wavelet transform
    window_size : int
        Size of the rolling window for volatility calculation
    threshold_multiplier : float
        Multiplier for the volatility threshold
        
    Returns:
    --------
    jumps : list of dict
        List of detected jumps with their properties
    """
    # Ensure we have returns
    if analyzer.returns is None:
        if np.all(analyzer.data > 0):
            analyzer.returns = np.diff(np.log(analyzer.data))
        else:
            analyzer.returns = np.diff(analyzer.data)
    
    returns = analyzer.returns
    n = len(returns)
    
    # Calculate rolling volatility
    rolling_std = np.zeros(n)
    for i in range(n):
        if i < window_size:
            # For the beginning, use available data
            rolling_std[i] = np.std(returns[:i+1]) if i > 0 else 0
        else:
            # Normal rolling window
            rolling_std[i] = np.std(returns[i-window_size+1:i+1])
    
    # Avoid division by zero
    rolling_std = np.maximum(rolling_std, 1e-8)
    
    # Calculate normalized returns
    normalized_returns = np.abs(returns) / rolling_std
    
    # Find returns exceeding the threshold
    jump_indices = np.where(normalized_returns > threshold_multiplier)[0]
    
    # Group consecutive jumps
    jumps = []
    if len(jump_indices) == 0:
        return jumps
    
    current_jump = [jump_indices[0]]
    
    for i in range(1, len(jump_indices)):
        if jump_indices[i] - jump_indices[i-1] <= 1:
            current_jump.append(jump_indices[i])
        else:
            # Process current jump
            jumps.append({
                'indices': current_jump,
                'start': current_jump[0],
                'end': current_jump[-1],
                'size': np.mean(np.abs(returns[current_jump])),
                'normalized_size': np.mean(normalized_returns[current_jump]),
                'returns': returns[current_jump],
                'volatility_ratio': np.mean(normalized_returns[current_jump]),
                'duration': len(current_jump)
            })
            current_jump = [jump_indices[i]]
    
    # Add the last jump
    if current_jump:
        jumps.append({
            'indices': current_jump,
            'start': current_jump[0],
            'end': current_jump[-1],
            'size': np.mean(np.abs(returns[current_jump])),
            'normalized_size': np.mean(normalized_returns[current_jump]),
            'returns': returns[current_jump],
            'volatility_ratio': np.mean(normalized_returns[current_jump]),
            'duration': len(current_jump)
        })
    
    # Sort jumps by normalized size
    jumps.sort(key=lambda x: x['normalized_size'], reverse=True)
    
    return jumps

def combined_jump_detection(analyzer, methods=['adaptive', 'multi_scale', 'volatility']):
    """
    Combine multiple jump detection methods.
    
    Parameters:
    -----------
    analyzer : FinancialWaveletAnalyzer
        Analyzer object with completed wavelet transform
    methods : list of str
        Jump detection methods to use
        
    Returns:
    --------
    jumps : list of dict
        Combined list of detected jumps with their properties
    """
    all_jumps = []
    
    if 'adaptive' in methods:
        adaptive_jumps = adaptive_threshold_jump_detection(analyzer, base_threshold=2.5)
        for jump in adaptive_jumps:
            jump['method'] = 'adaptive'
        all_jumps.extend(adaptive_jumps)
    
    if 'multi_scale' in methods:
        multi_scale_jumps = multi_scale_jump_detection(analyzer, scales_to_use=3)
        for jump in multi_scale_jumps:
            jump['method'] = 'multi_scale'
        all_jumps.extend(multi_scale_jumps)
    
    if 'volatility' in methods:
        volatility_jumps = volatility_aware_jump_detection(analyzer)
        for jump in volatility_jumps:
            jump['method'] = 'volatility'
        all_jumps.extend(volatility_jumps)
    
    # Merge overlapping jumps
    if all_jumps:
        # Sort by start time
        all_jumps.sort(key=lambda x: x['start'])
        
        merged_jumps = [all_jumps[0]]
        
        for i in range(1, len(all_jumps)):
            current = all_jumps[i]
            previous = merged_jumps[-1]
            
            # Check if current jump overlaps with previous
            if current['start'] <= previous['end'] + 1:
                # Merge jumps
                end = max(previous['end'], current['end'])
                merged_indices = list(set(previous['indices'] + current['indices']))
                merged_indices.sort()
                
                # Update the previous jump
                previous['end'] = end
                previous['indices'] = merged_indices
                previous['methods'] = [previous.get('method', 'unknown'), 
                                      current.get('method', 'unknown')]
                previous['size'] = max(previous.get('size', 0), current.get('size', 0))
                
                # Remove method-specific field to avoid confusion
                previous.pop('method', None)
            else:
                # Add as a new jump
                merged_jumps.append(current)
        
        all_jumps = merged_jumps
    
    # Sort by size/importance
    if all_jumps:
        all_jumps.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    return all_jumps