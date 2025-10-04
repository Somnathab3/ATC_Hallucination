"""
Module Name: ingest.py
Description: Data ingestion and harmonization pipeline for trajectory and hallucination data.
Author: Som
Date: 2025-10-04

Handles loading and preprocessing of:
- Per-episode trajectory CSVs
- Per-test hallucination JSON/CSV results
- Aggregate shift summary data
"""

import pandas as pd
import numpy as np
import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union

def load_trajectories(path_pattern="results/trajectory_*.csv", 
                     results_dir="results", 
                     add_metadata=True):
    """
    Load and concatenate trajectory CSV files.
    
    Args:
        path_pattern: Glob pattern for trajectory files
        results_dir: Base results directory
        add_metadata: Whether to extract metadata from filenames
        
    Returns:
        pandas.DataFrame: Concatenated trajectory data
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(path_pattern):
        path_pattern = os.path.join(results_dir, path_pattern)
    
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        print(f"Warning: No trajectory files found matching pattern: {path_pattern}")
        return pd.DataFrame()
    
    dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if add_metadata:
                # Extract metadata from filename
                filename = os.path.basename(csv_file)
                metadata = _extract_metadata_from_filename(filename)
                
                # Add metadata columns
                for key, value in metadata.items():
                    df[key] = value
                
                # Add source file
                df['source_file'] = filename
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue
    
    if not dataframes:
        return pd.DataFrame()
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Standardize column names
    combined_df = _standardize_columns(combined_df)
    
    return combined_df

def load_hallucination_results(path_pattern="results/hallucination_*.json",
                              results_dir="results"):
    """
    Load hallucination analysis results from JSON files.
    
    Args:
        path_pattern: Glob pattern for hallucination files
        results_dir: Base results directory
        
    Returns:
        list: List of hallucination result dictionaries
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(path_pattern):
        path_pattern = os.path.join(results_dir, path_pattern)
    
    json_files = glob.glob(path_pattern)
    
    if not json_files:
        # Try CSV files as alternative
        csv_pattern = path_pattern.replace('.json', '.csv')
        return load_hallucination_csv(csv_pattern)
    
    results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Add file metadata
                data['source_file'] = os.path.basename(json_file)
                
                # Extract metadata from filename
                metadata = _extract_metadata_from_filename(os.path.basename(json_file))
                data.update(metadata)
                
                results.append(data)
                
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    return results

def load_hallucination_csv(path_pattern="results/hallucination_*.csv",
                          results_dir="results"):
    """
    Load hallucination results from CSV files.
    
    Args:
        path_pattern: Glob pattern for CSV files
        results_dir: Base results directory
        
    Returns:
        pandas.DataFrame: Combined hallucination data
    """
    if not os.path.isabs(path_pattern):
        path_pattern = os.path.join(results_dir, path_pattern)
    
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        print(f"Warning: No hallucination CSV files found: {path_pattern}")
        return pd.DataFrame()
    
    dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Add metadata
            filename = os.path.basename(csv_file)
            metadata = _extract_metadata_from_filename(filename)
            
            for key, value in metadata.items():
                df[key] = value
            
            df['source_file'] = filename
            dataframes.append(df)
            
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue
    
    if not dataframes:
        return pd.DataFrame()
    
    return pd.concat(dataframes, ignore_index=True)

def load_shift_summary(summary_file="results/shift_test_summary.csv"):
    """
    Load shift test summary data.
    
    Args:
        summary_file: Path to summary CSV file
        
    Returns:
        pandas.DataFrame: Summary data
    """
    try:
        df = pd.read_csv(summary_file)
        return _standardize_columns(df)
    except Exception as e:
        print(f"Warning: Failed to load summary file {summary_file}: {e}")
        return pd.DataFrame()

def _extract_metadata_from_filename(filename):
    """
    Extract metadata from standardized filename patterns.
    
    Args:
        filename: File basename
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {}
    
    # Remove extension
    name_parts = os.path.splitext(filename)[0].split('_')
    
    # Common patterns
    if 'trajectory' in filename:
        metadata['data_type'] = 'trajectory'
    elif 'hallucination' in filename:
        metadata['data_type'] = 'hallucination'
    elif 'summary' in filename:
        metadata['data_type'] = 'summary'
    
    # Extract scenario names
    scenarios = ['parallel', 'head_on', 'converging', 'crossing', 't_formation']
    for scenario in scenarios:
        if scenario in filename.lower():
            metadata['scenario'] = scenario
            break
    
    # Extract shift types
    shift_types = ['heading', 'speed', 'position', 'lateral', 'vertical']
    for shift_type in shift_types:
        if shift_type in filename.lower():
            metadata['shift_type'] = shift_type
            break
    
    # Extract episode numbers
    for part in name_parts:
        if part.startswith('ep') and part[2:].isdigit():
            metadata['episode_id'] = int(part[2:])
        elif part.startswith('episode') and part[7:].isdigit():
            metadata['episode_id'] = int(part[7:])
    
    # Extract shift magnitudes (look for numbers)
    for part in name_parts:
        try:
            # Check if it's a float
            if '.' in part:
                magnitude = float(part)
                if 0 <= magnitude <= 100:  # Reasonable magnitude range
                    metadata['shift_magnitude'] = magnitude
            # Check if it's an integer
            elif part.isdigit():
                magnitude = int(part)
                if 0 <= magnitude <= 100:
                    metadata['shift_magnitude'] = magnitude
        except ValueError:
            continue
    
    # Extract timestamps
    for part in name_parts:
        if len(part) == 8 and part.isdigit():  # YYYYMMDD format
            metadata['date'] = part
        elif len(part) == 6 and part.isdigit():  # HHMMSS format
            metadata['time'] = part
    
    return metadata

def _standardize_columns(df):
    """
    Standardize column names across different data sources.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with standardized columns
    """
    # Common column name mappings
    column_mappings = {
        'lat': 'lat_deg',
        'latitude': 'lat_deg',
        'lon': 'lon_deg', 
        'longitude': 'lon_deg',
        'heading': 'hdg_deg',
        'speed': 'tas_kt',
        'true_airspeed': 'tas_kt',
        'agent': 'agent_id',
        'aircraft': 'agent_id',
        'time': 'sim_time_s',
        'simulation_time': 'sim_time_s',
        'step': 'step_idx',
        'timestep': 'step_idx',
        'minimum_separation': 'min_separation_nm',
        'min_sep': 'min_separation_nm',
        'conflict': 'conflict_flag',
        'alert': 'predicted_alert',
        'prediction': 'predicted_alert'
    }
    
    # Apply mappings
    df = df.rename(columns=column_mappings)
    
    # Ensure consistent data types
    type_mappings = {
        'step_idx': 'int64',
        'sim_time_s': 'float64',
        'lat_deg': 'float64',
        'lon_deg': 'float64',
        'hdg_deg': 'float64',
        'tas_kt': 'float64',
        'min_separation_nm': 'float64',
        'conflict_flag': 'int64',
        'predicted_alert': 'int64'
    }
    
    for col, dtype in type_mappings.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {col} to {dtype}")
    
    return df

def harmonize_hallucination_data(trajectory_df, hallucination_results):
    """
    Merge trajectory data with hallucination results.
    
    Args:
        trajectory_df: Trajectory DataFrame
        hallucination_results: List of hallucination result dicts or DataFrame
        
    Returns:
        pandas.DataFrame: Merged DataFrame with hallucination flags
    """
    if isinstance(hallucination_results, list):
        # Convert list of dicts to DataFrame
        if not hallucination_results:
            return trajectory_df
        
        hall_df = pd.DataFrame(hallucination_results)
    else:
        hall_df = hallucination_results.copy()
    
    # Initialize hallucination columns in trajectory data
    for col in ['fp', 'fn', 'tp', 'tn']:
        if col not in trajectory_df.columns:
            trajectory_df[col] = 0
    
    # Merge based on available keys
    merge_keys = []
    for key in ['episode_id', 'step_idx', 'agent_id', 'sim_time_s']:
        if key in trajectory_df.columns and key in hall_df.columns:
            merge_keys.append(key)
    
    if not merge_keys:
        print("Warning: No common keys found for merging hallucination data")
        return trajectory_df
    
    # Perform merge
    try:
        merged_df = pd.merge(trajectory_df, hall_df, on=merge_keys, how='left', suffixes=('', '_hall'))
        
        # Update hallucination flags
        for col in ['fp', 'fn', 'tp', 'tn']:
            if col + '_hall' in merged_df.columns:
                merged_df[col] = merged_df[col + '_hall'].fillna(merged_df[col])
                merged_df.drop(columns=[col + '_hall'], inplace=True)
        
        return merged_df
        
    except Exception as e:
        print(f"Warning: Failed to merge hallucination data: {e}")
        return trajectory_df

def load_complete_dataset(results_dir="results", 
                         trajectory_pattern="trajectory_*.csv",
                         hallucination_pattern="hallucination_*.json",
                         summary_file="shift_test_summary.csv"):
    """
    Load complete dataset with all components.
    
    Args:
        results_dir: Base results directory
        trajectory_pattern: Pattern for trajectory files
        hallucination_pattern: Pattern for hallucination files  
        summary_file: Summary file name
        
    Returns:
        dict: Complete dataset with trajectories, hallucinations, and summary
    """
    print("Loading trajectory data...")
    trajectories = load_trajectories(trajectory_pattern, results_dir)
    
    print("Loading hallucination results...")
    hallucinations = load_hallucination_results(hallucination_pattern, results_dir)
    
    print("Loading summary data...")
    summary = load_shift_summary(os.path.join(results_dir, summary_file))
    
    print("Harmonizing data...")
    if not trajectories.empty and hallucinations:
        trajectories = harmonize_hallucination_data(trajectories, hallucinations)
    
    dataset = {
        'trajectories': trajectories,
        'hallucinations': hallucinations,
        'summary': summary,
        'metadata': {
            'n_trajectory_records': len(trajectories),
            'n_hallucination_records': len(hallucinations) if isinstance(hallucinations, list) else len(hallucinations),
            'n_summary_records': len(summary),
            'trajectory_columns': list(trajectories.columns) if not trajectories.empty else [],
            'summary_columns': list(summary.columns) if not summary.empty else []
        }
    }
    
    print(f"Dataset loaded: {dataset['metadata']}")
    return dataset

def export_harmonized_data(dataset, output_dir="processed_data"):
    """
    Export harmonized dataset to standardized format.
    
    Args:
        dataset: Dataset dictionary from load_complete_dataset
        output_dir: Output directory for processed files
        
    Returns:
        dict: Paths to exported files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    # Export trajectories
    if not dataset['trajectories'].empty:
        traj_file = os.path.join(output_dir, "trajectories_harmonized.csv")
        dataset['trajectories'].to_csv(traj_file, index=False)
        exported_files['trajectories'] = traj_file
    
    # Export summary
    if not dataset['summary'].empty:
        summary_file = os.path.join(output_dir, "summary_harmonized.csv")
        dataset['summary'].to_csv(summary_file, index=False)
        exported_files['summary'] = summary_file
    
    # Export metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(dataset['metadata'], f, indent=2)
    exported_files['metadata'] = metadata_file
    
    print(f"Exported files: {exported_files}")
    return exported_files