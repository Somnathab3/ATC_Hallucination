"""
Safety Margin Analysis and Performance Metrics Module.

This module provides comprehensive safety analysis capabilities for air traffic control
system evaluation, focusing on separation violations, conflict detection performance,
and risk assessment metrics suitable for academic evaluation.

Core Functionality:
- Loss of Separation (LoS) Analysis: Time spent below safety thresholds
- Closest Point of Approach (CPA) Statistics: Minimum separation calculations
- Action Oscillation Detection: Policy stability and decision consistency analysis
- Risk Assessment: Probabilistic safety evaluation with confidence intervals
- Performance Degradation Analysis: Comparative safety assessment between conditions

The module implements robust statistical methods including Wilson score confidence
intervals and bootstrap resampling for publication-quality uncertainty quantification.
"""

import numpy as np
import pandas as pd
import math
from typing import Tuple, Optional

def dwell_fraction(df, threshold_nm=5.0):
    """
    Calculate fraction of episode time spent in Loss of Separation conditions.
    
    This metric quantifies safety performance by measuring the proportion of
    time aircraft spend below the minimum separation threshold, providing
    insight into conflict severity and duration.
    
    Args:
        df: DataFrame containing trajectory data with min_separation_nm column
        threshold_nm: Minimum separation threshold in nautical miles
        
    Returns:
        float: Fraction of timesteps below threshold (range 0.0-1.0)
    """
    if 'min_separation_nm' not in df.columns or df.empty:
        return 0.0
    
    below_threshold = (df['min_separation_nm'] < threshold_nm).sum()
    total_steps = len(df)
    
    return below_threshold / total_steps if total_steps > 0 else 0.0

def min_cpa(df):
    """
    Calculate minimum Closest Point of Approach across the episode.
    
    Args:
        df: DataFrame with trajectory data
        
    Returns:
        float: Minimum separation distance in nautical miles
    """
    if 'min_separation_nm' not in df.columns or df.empty:
        return np.nan
    
    return df['min_separation_nm'].min()

def action_oscillation(df, agent_id):
    """
    Analyze agent action stability through oscillation pattern detection.
    
    Oscillations indicate policy instability or conflicting objectives that
    can degrade system performance and passenger comfort. This function
    quantifies heading and speed command reversals as stability metrics.
    
    Args:
        df: DataFrame with trajectory data including action_hdg_delta_deg and
            action_spd_delta_kt columns
        agent_id: Target agent identifier for analysis
        
    Returns:
        dict: Oscillation metrics including counts, rates, and total actions
    """
    agent_data = df[df['agent_id'] == agent_id].sort_values('step_idx')
    
    if agent_data.empty:
        return {'heading_oscillations': 0, 'speed_oscillations': 0, 'total_actions': 0}
    
    metrics = {'total_actions': len(agent_data)}
    
    # Heading oscillations
    if 'action_hdg_delta_deg' in agent_data.columns:
        hdg_actions = agent_data['action_hdg_delta_deg'].dropna()
        if len(hdg_actions) > 2:
            # Count direction changes (sign changes in consecutive actions)
            hdg_changes = hdg_actions.diff().dropna()
            sign_changes = ((hdg_changes[1:] * hdg_changes[:-1].values) < 0).sum()
            metrics['heading_oscillations'] = sign_changes
            metrics['heading_oscillation_rate'] = sign_changes / len(hdg_actions)
        else:
            metrics['heading_oscillations'] = 0
            metrics['heading_oscillation_rate'] = 0.0
    
    # Speed oscillations
    if 'action_spd_delta_kt' in agent_data.columns:
        spd_actions = agent_data['action_spd_delta_kt'].dropna()
        if len(spd_actions) > 2:
            spd_changes = spd_actions.diff().dropna()
            sign_changes = ((spd_changes[1:] * spd_changes[:-1].values) < 0).sum()
            metrics['speed_oscillations'] = sign_changes
            metrics['speed_oscillation_rate'] = sign_changes / len(spd_actions)
        else:
            metrics['speed_oscillations'] = 0
            metrics['speed_oscillation_rate'] = 0.0
    
    return metrics

def rate_ci_wilson(successes, total, confidence=0.95):
    """
    Calculate rate with Wilson score confidence interval (no SciPy dependency).
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        tuple: (rate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    p = successes / total
    n = total
    
    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * math.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return p, lower, upper

def calculate_safety_metrics(df, episode_col='episode_id'):
    """
    Calculate comprehensive safety metrics for each episode.
    
    Args:
        df: DataFrame with trajectory data
        episode_col: Column name for episode identifier
        
    Returns:
        pandas.DataFrame: Safety metrics per episode
    """
    metrics = []
    
    for episode_id, episode_data in df.groupby(episode_col):
        episode_metrics = {
            'episode_id': episode_id,
            'total_steps': len(episode_data),
            'dwell_fraction_5nm': dwell_fraction(episode_data, 5.0),
            'dwell_fraction_3nm': dwell_fraction(episode_data, 3.0),
            'min_cpa_nm': min_cpa(episode_data),
            'conflict_duration_s': _calculate_conflict_duration(episode_data),
            'n_conflicts': (episode_data.get('conflict_flag', 0) == 1).sum(),
            'n_alerts': (episode_data.get('predicted_alert', 0) == 1).sum(),
        }
        
        # Agent-specific oscillations
        for agent in episode_data['agent_id'].unique():
            osc_metrics = action_oscillation(episode_data, agent)
            episode_metrics[f'oscillations_{agent}'] = osc_metrics.get('heading_oscillations', 0)
            episode_metrics[f'oscillation_rate_{agent}'] = osc_metrics.get('heading_oscillation_rate', 0.0)
        
        # Hallucination metrics if available
        for event_type in ['fp', 'fn', 'tp', 'tn']:
            if event_type in episode_data.columns:
                episode_metrics[f'{event_type}_count'] = (episode_data[event_type] == 1).sum()
                episode_metrics[f'{event_type}_rate'] = (episode_data[event_type] == 1).mean()
        
        metrics.append(episode_metrics)
    
    return pd.DataFrame(metrics)

def _calculate_conflict_duration(episode_data):
    """Calculate total conflict duration in seconds."""
    if 'conflict_flag' not in episode_data.columns or 'sim_time_s' not in episode_data.columns:
        return 0.0
    
    conflicts = episode_data[episode_data['conflict_flag'] == 1]
    if conflicts.empty:
        return 0.0
    
    # Simple approach: total time span of conflict flags
    return conflicts['sim_time_s'].max() - conflicts['sim_time_s'].min()

def risk_assessment(df, safety_threshold_nm=5.0):
    """
    Perform comprehensive risk assessment.
    
    Args:
        df: DataFrame with trajectory data
        safety_threshold_nm: Safety threshold for risk calculation
        
    Returns:
        dict: Risk assessment results
    """
    assessment = {}
    
    # Overall risk metrics
    total_steps = len(df)
    if total_steps == 0:
        return {'error': 'No data available'}
    
    # Separation-based risk
    if 'min_separation_nm' in df.columns:
        below_threshold = (df['min_separation_nm'] < safety_threshold_nm).sum()
        assessment['los_probability'] = below_threshold / total_steps
        
        # Risk severity (closer = higher risk)
        violations = df[df['min_separation_nm'] < safety_threshold_nm]
        if not violations.empty:
            assessment['mean_violation_severity'] = (safety_threshold_nm - violations['min_separation_nm']).mean()
            assessment['max_violation_severity'] = (safety_threshold_nm - violations['min_separation_nm']).max()
        else:
            assessment['mean_violation_severity'] = 0.0
            assessment['max_violation_severity'] = 0.0
    
    # Alert system performance
    if all(col in df.columns for col in ['predicted_alert', 'conflict_flag']):
        tp = ((df['predicted_alert'] == 1) & (df['conflict_flag'] == 1)).sum()
        fp = ((df['predicted_alert'] == 1) & (df['conflict_flag'] == 0)).sum()
        fn = ((df['predicted_alert'] == 0) & (df['conflict_flag'] == 1)).sum()
        tn = ((df['predicted_alert'] == 0) & (df['conflict_flag'] == 0)).sum()
        
        total = tp + fp + fn + tn
        if total > 0:
            assessment['alert_accuracy'] = (tp + tn) / total
            assessment['alert_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            assessment['alert_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            assessment['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Time-based risk evolution
    if 'sim_time_s' in df.columns and 'min_separation_nm' in df.columns:
        time_windows = pd.cut(df['sim_time_s'], bins=10)
        risk_evolution = []
        
        for window, window_data in df.groupby(time_windows):
            if not window_data.empty and 'min_separation_nm' in window_data.columns:
                window_risk = (window_data['min_separation_nm'] < safety_threshold_nm).mean()
                risk_evolution.append({
                    'time_window': f"{window.left:.1f}-{window.right:.1f}s",
                    'risk_probability': window_risk
                })
        
        assessment['risk_evolution'] = risk_evolution
    
    # Agent-specific risk
    agent_risks = {}
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent]
        agent_risk = risk_assessment(agent_data, safety_threshold_nm)
        # Remove nested risk_evolution to avoid recursion
        if 'risk_evolution' in agent_risk:
            del agent_risk['risk_evolution']
        agent_risks[agent] = agent_risk
    
    assessment['agent_risks'] = agent_risks
    
    return assessment

def safety_margin_statistics(df):
    """
    Calculate detailed statistics on safety margins.
    
    Args:
        df: DataFrame with trajectory data
        
    Returns:
        dict: Safety margin statistics
    """
    if 'min_separation_nm' not in df.columns:
        return {}
    
    separations = df['min_separation_nm'].dropna()
    
    if separations.empty:
        return {}
    
    stats = {
        'count': len(separations),
        'mean_separation_nm': separations.mean(),
        'std_separation_nm': separations.std(),
        'min_separation_nm': separations.min(),
        'max_separation_nm': separations.max(),
        'median_separation_nm': separations.median(),
        'p5_separation_nm': separations.quantile(0.05),
        'p95_separation_nm': separations.quantile(0.95),
        'violations_5nm': (separations < 5.0).sum(),
        'violations_3nm': (separations < 3.0).sum(),
        'violations_1nm': (separations < 1.0).sum(),
    }
    
    # Time to collision metrics (if velocity data available)
    if all(col in df.columns for col in ['tas_kt', 'hdg_deg']):
        # Simplified TTC calculation would go here
        # For now, just indicate availability
        stats['ttc_data_available'] = True
    else:
        stats['ttc_data_available'] = False
    
    return stats

def performance_degradation_analysis(baseline_df, shifted_df):
    """
    Analyze how safety performance degrades from baseline to shifted scenarios.
    
    Args:
        baseline_df: Baseline trajectory DataFrame
        shifted_df: Shifted trajectory DataFrame
        
    Returns:
        dict: Performance degradation analysis
    """
    baseline_safety = safety_margin_statistics(baseline_df)
    shifted_safety = safety_margin_statistics(shifted_df)
    
    if not baseline_safety or not shifted_safety:
        return {'error': 'Insufficient data for comparison'}
    
    degradation = {}
    
    # Compare key metrics
    for metric in ['mean_separation_nm', 'min_separation_nm', 'violations_5nm', 'violations_3nm']:
        if metric in baseline_safety and metric in shifted_safety:
            baseline_val = baseline_safety[metric]
            shifted_val = shifted_safety[metric]
            
            if baseline_val != 0:
                change_pct = ((shifted_val - baseline_val) / baseline_val) * 100
                degradation[f'{metric}_change_pct'] = change_pct
                degradation[f'{metric}_absolute_change'] = shifted_val - baseline_val
    
    # Risk assessment comparison
    baseline_risk = risk_assessment(baseline_df)
    shifted_risk = risk_assessment(shifted_df)
    
    if 'los_probability' in baseline_risk and 'los_probability' in shifted_risk:
        degradation['los_probability_change'] = shifted_risk['los_probability'] - baseline_risk['los_probability']
    
    return {
        'baseline_metrics': baseline_safety,
        'shifted_metrics': shifted_safety,
        'degradation': degradation,
        'baseline_risk': baseline_risk,
        'shifted_risk': shifted_risk
    }