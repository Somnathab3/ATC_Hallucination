"""
Module Name: analysis_similarity.py
Description: Trajectory similarity analysis using Dynamic Time Warping (DTW).
Author: Som
Date: 2025-10-04

Provides trajectory comparison capabilities for evaluating distribution shift impact
on aircraft behavior patterns using DTW for robust temporal alignment.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def simple_dtw_distance(x, y):
    """
    Compute Dynamic Time Warping distance between two time series.
    
    Args:
        x: First time series as numpy array
        y: Second time series as numpy array
        
    Returns:
        float: DTW distance (lower values = greater similarity)
    """
    n, m = len(x), len(y)
    
    # Create cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            # Take minimum of three possible paths
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]

def dtw_traj_nm(baseline_df, shifted_df, agent_id, metric='position'):
    """
    Calculate DTW distance between baseline and shifted trajectories.
    
    Args:
        baseline_df: Baseline trajectory DataFrame
        shifted_df: Shifted trajectory DataFrame
        agent_id: Agent identifier
        metric: Type of trajectory to compare ('position', 'heading', 'speed')
        
    Returns:
        float: DTW distance in appropriate units
    """
    # Filter for specific agent
    base_agent = baseline_df[baseline_df['agent_id'] == agent_id].sort_values('step_idx')
    shift_agent = shifted_df[shifted_df['agent_id'] == agent_id].sort_values('step_idx')
    
    if base_agent.empty or shift_agent.empty:
        return np.nan
    
    if metric == 'position':
        # Calculate position-based DTW using lat/lon
        base_positions = np.column_stack((base_agent['lat_deg'].values, base_agent['lon_deg'].values))
        shift_positions = np.column_stack((shift_agent['lat_deg'].values, shift_agent['lon_deg'].values))
        
        # Convert to distances from first point (simplified)
        base_distances = np.cumsum(np.concatenate([[0], np.sqrt(np.sum(np.diff(base_positions, axis=0)**2, axis=1))]))
        shift_distances = np.cumsum(np.concatenate([[0], np.sqrt(np.sum(np.diff(shift_positions, axis=0)**2, axis=1))]))
        
        # Convert to nautical miles (rough approximation)
        base_distances_nm = base_distances * 60  # degrees to NM
        shift_distances_nm = shift_distances * 60
        
        return simple_dtw_distance(base_distances_nm, shift_distances_nm)
    
    elif metric == 'heading':
        if 'hdg_deg' in base_agent.columns and 'hdg_deg' in shift_agent.columns:
            base_hdg = base_agent['hdg_deg'].values
            shift_hdg = shift_agent['hdg_deg'].values
            return simple_dtw_distance(base_hdg, shift_hdg)
        else:
            return np.nan
    
    elif metric == 'speed':
        if 'tas_kt' in base_agent.columns and 'tas_kt' in shift_agent.columns:
            base_speed = base_agent['tas_kt'].values
            shift_speed = shift_agent['tas_kt'].values
            return simple_dtw_distance(base_speed, shift_speed)
        else:
            return np.nan
    
    return np.nan

def trajectory_deviation_metrics(baseline_df, shifted_df, agent_id):
    """
    Calculate comprehensive trajectory deviation metrics.
    
    Args:
        baseline_df: Baseline trajectory DataFrame
        shifted_df: Shifted trajectory DataFrame
        agent_id: Agent identifier
        
    Returns:
        dict: Dictionary of deviation metrics
    """
    base_agent = baseline_df[baseline_df['agent_id'] == agent_id].sort_values('step_idx')
    shift_agent = shifted_df[shifted_df['agent_id'] == agent_id].sort_values('step_idx')
    
    if base_agent.empty or shift_agent.empty:
        return {}
    
    metrics = {}
    
    # Align trajectories by time or step (use minimum length)
    min_len = min(len(base_agent), len(shift_agent))
    base_aligned = base_agent.iloc[:min_len]
    shift_aligned = shift_agent.iloc[:min_len]
    
    # Position deviation
    if all(col in base_aligned.columns for col in ['lat_deg', 'lon_deg']):
        lat_diff = shift_aligned['lat_deg'].values - base_aligned['lat_deg'].values
        lon_diff = shift_aligned['lon_deg'].values - base_aligned['lon_deg'].values
        
        # Convert to approximate distance in NM
        position_deviation_nm = np.sqrt(lat_diff**2 + lon_diff**2) * 60  # rough conversion
        
        metrics['max_position_deviation_nm'] = np.max(position_deviation_nm)
        metrics['mean_position_deviation_nm'] = np.mean(position_deviation_nm)
        metrics['final_position_deviation_nm'] = position_deviation_nm[-1] if len(position_deviation_nm) > 0 else 0
    
    # Heading deviation
    if 'hdg_deg' in base_aligned.columns and 'hdg_deg' in shift_aligned.columns:
        hdg_diff = _angle_difference(shift_aligned['hdg_deg'].values, base_aligned['hdg_deg'].values)
        
        metrics['max_heading_deviation_deg'] = np.max(np.abs(hdg_diff))
        metrics['mean_heading_deviation_deg'] = np.mean(np.abs(hdg_diff))
        metrics['final_heading_deviation_deg'] = abs(hdg_diff[-1]) if len(hdg_diff) > 0 else 0
    
    # Speed deviation
    if 'tas_kt' in base_aligned.columns and 'tas_kt' in shift_aligned.columns:
        speed_diff = shift_aligned['tas_kt'].values - base_aligned['tas_kt'].values
        
        metrics['max_speed_deviation_kt'] = np.max(np.abs(speed_diff))
        metrics['mean_speed_deviation_kt'] = np.mean(np.abs(speed_diff))
        metrics['final_speed_deviation_kt'] = abs(speed_diff[-1]) if len(speed_diff) > 0 else 0
    
    # DTW distances
    metrics['dtw_position'] = dtw_traj_nm(baseline_df, shifted_df, agent_id, 'position')
    metrics['dtw_heading'] = dtw_traj_nm(baseline_df, shifted_df, agent_id, 'heading')
    metrics['dtw_speed'] = dtw_traj_nm(baseline_df, shifted_df, agent_id, 'speed')
    
    return metrics

def _angle_difference(angle1, angle2):
    """Calculate the difference between two angles in degrees, handling wrap-around."""
    diff = angle1 - angle2
    return ((diff + 180) % 360) - 180

def trajectory_similarity_matrix(baseline_df, shifted_df, agents=None, metric='position'):
    """
    Create similarity matrix between all agent trajectories.
    
    Args:
        baseline_df: Baseline trajectory DataFrame
        shifted_df: Shifted trajectory DataFrame
        agents: List of agents to analyze (auto-detected if None)
        metric: Similarity metric to use
        
    Returns:
        pandas.DataFrame: Similarity matrix
    """
    if agents is None:
        agents = sorted(set(baseline_df['agent_id'].unique()) & set(shifted_df['agent_id'].unique()))
    
    n_agents = len(agents)
    similarity_matrix = np.zeros((n_agents, n_agents))
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i == j:
                similarity_matrix[i, j] = 0  # Same agent baseline vs shifted
                similarity_matrix[i, j] = dtw_traj_nm(baseline_df, shifted_df, agent1, metric)
            else:
                # Cross-agent similarity (baseline agent1 vs shifted agent2)
                base_agent1 = baseline_df[baseline_df['agent_id'] == agent1]
                shift_agent2 = shifted_df[shifted_df['agent_id'] == agent2]
                
                if not base_agent1.empty and not shift_agent2.empty:
                    # Create temporary DataFrames for DTW calculation
                    temp_baseline = base_agent1.copy()
                    temp_shifted = shift_agent2.copy()
                    temp_shifted['agent_id'] = agent1  # Trick DTW function
                    
                    similarity_matrix[i, j] = dtw_traj_nm(temp_baseline, temp_shifted, agent1, metric)
                else:
                    similarity_matrix[i, j] = np.nan
    
    return pd.DataFrame(similarity_matrix, index=agents, columns=agents)

def analyze_trajectory_clustering(baseline_df, shifted_df):
    """
    Analyze how much trajectories cluster or disperse under shifts.
    
    Args:
        baseline_df: Baseline trajectory DataFrame
        shifted_df: Shifted trajectory DataFrame
        
    Returns:
        dict: Clustering analysis results
    """
    agents = sorted(set(baseline_df['agent_id'].unique()) & set(shifted_df['agent_id'].unique()))
    
    results = {
        'agent_deviations': {},
        'cross_agent_similarity': {},
        'overall_dispersion': {}
    }
    
    # Analyze each agent's deviation
    for agent in agents:
        metrics = trajectory_deviation_metrics(baseline_df, shifted_df, agent)
        results['agent_deviations'][agent] = metrics
    
    # Cross-agent similarity analysis
    for metric in ['position', 'heading', 'speed']:
        sim_matrix = trajectory_similarity_matrix(baseline_df, shifted_df, agents, metric)
        results['cross_agent_similarity'][metric] = {
            'similarity_matrix': sim_matrix,
            'mean_similarity': sim_matrix.mean().mean(),
            'std_similarity': sim_matrix.stack().std()
        }
    
    # Overall dispersion metrics
    all_deviations = []
    for agent_metrics in results['agent_deviations'].values():
        if 'mean_position_deviation_nm' in agent_metrics:
            all_deviations.append(agent_metrics['mean_position_deviation_nm'])
    
    if all_deviations:
        results['overall_dispersion'] = {
            'mean_deviation_nm': np.mean(all_deviations),
            'std_deviation_nm': np.std(all_deviations),
            'max_deviation_nm': np.max(all_deviations),
            'coefficient_of_variation': np.std(all_deviations) / np.mean(all_deviations)
        }
    
    return results