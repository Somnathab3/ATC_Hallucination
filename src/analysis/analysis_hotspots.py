"""
Spatial Hotspot Analysis for Conflict and Hallucination Events.

This module implements advanced spatial clustering techniques to identify geographic
regions where conflicts and hallucination events concentrate, providing insights
into airspace design and policy failure modes.

Core Functionality:
- DBSCAN Clustering: Density-based spatial clustering of event locations
- Hallucination Hotspots: Geographic concentration of false positives and false negatives
- Conflict Zone Identification: Areas with frequent Loss of Separation events
- Statistical Analysis: Cluster characterization with geometric and temporal metrics

Applications:
- Identifying problematic airspace regions requiring design attention
- Understanding geographic bias in conflict detection algorithms
- Academic analysis of spatial patterns in safety-critical aviation systems
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in nautical miles."""
    R_nm = 3440.065  # Earth radius in nautical miles
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    
    return 2 * R_nm * np.arcsin(np.sqrt(a))

def cluster_fp_fn(df, kind="fp", eps_nm=3.0, min_samples=6):
    """
    Apply DBSCAN clustering to hallucination events by geographic location.
    
    Uses density-based clustering to identify regions where hallucination events
    (false positives, false negatives, etc.) concentrate, enabling identification
    of systematic biases or problematic airspace regions.
    
    Args:
        df: DataFrame containing event data with lat_deg, lon_deg columns
        kind: Event type to cluster ("fp", "fn", "tp", "tn")
        eps_nm: Maximum distance between cluster members in nautical miles
        min_samples: Minimum events required to form a cluster
        
    Returns:
        pandas.DataFrame: Input DataFrame with cluster labels (-1 for noise)
    """
    # Filter for the specified event type
    events = df[df.get(kind, 0) == 1].copy()
    
    if events.empty:
        df['cluster'] = -1
        return df
    
    # Extract coordinates
    coords = events[['lat_deg', 'lon_deg']].values
    
    # Convert eps to approximate degrees (rough conversion)
    eps_deg = eps_nm / 60.0  # 1 degree ≈ 60 nautical miles
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps_deg, min_samples=min_samples, metric='haversine')
    
    # Fit and predict
    cluster_labels = clustering.fit_predict(np.radians(coords))
    
    # Add cluster labels to events
    events['cluster'] = cluster_labels
    
    # Merge back with original DataFrame
    df = df.copy()
    df['cluster'] = -1  # Default no cluster
    
    for idx, cluster in zip(events.index, cluster_labels):
        df.loc[idx, 'cluster'] = cluster
    
    return df

def analyze_hotspots(df, event_types=['fp', 'fn'], eps_nm=3.0, min_samples=5):
    """
    Comprehensive hotspot analysis for multiple event types.
    
    Args:
        df: DataFrame with event data
        event_types: List of event types to analyze
        eps_nm: Clustering distance threshold
        min_samples: Minimum samples per cluster
        
    Returns:
        dict: Analysis results for each event type
    """
    results = {}
    
    for event_type in event_types:
        if event_type not in df.columns:
            continue
            
        # Cluster events
        clustered_df = cluster_fp_fn(df, kind=event_type, eps_nm=eps_nm, min_samples=min_samples)
        
        # Analyze clusters
        clusters = clustered_df[clustered_df['cluster'] >= 0]
        
        if clusters.empty:
            results[event_type] = {
                'n_clusters': 0,
                'n_events': 0,
                'cluster_stats': pd.DataFrame()
            }
            continue
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in clusters['cluster'].unique():
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'n_events': len(cluster_data),
                'center_lat': cluster_data['lat_deg'].mean(),
                'center_lon': cluster_data['lon_deg'].mean(),
                'radius_nm': _calculate_cluster_radius(cluster_data),
                'agents_involved': cluster_data['agent_id'].nunique(),
                'time_span_s': cluster_data['sim_time_s'].max() - cluster_data['sim_time_s'].min(),
                'episodes_involved': cluster_data.get('episode_id', pd.Series()).nunique()
            }
            cluster_stats.append(stats)
        
        results[event_type] = {
            'n_clusters': len(cluster_stats),
            'n_events': len(clusters),
            'cluster_stats': pd.DataFrame(cluster_stats),
            'clustered_df': clustered_df
        }
    
    return results

def _calculate_cluster_radius(cluster_data):
    """Calculate radius of a cluster in nautical miles."""
    if len(cluster_data) <= 1:
        return 0.0
    
    center_lat = cluster_data['lat_deg'].mean()
    center_lon = cluster_data['lon_deg'].mean()
    
    distances = []
    for _, row in cluster_data.iterrows():
        dist = haversine_distance(center_lat, center_lon, row['lat_deg'], row['lon_deg'])
        distances.append(dist)
    
    return np.max(distances)

def identify_conflict_zones(df, conflict_threshold_nm=5.0, eps_nm=2.0, min_samples=3):
    """
    Identify zones where conflicts frequently occur.
    
    Args:
        df: DataFrame with trajectory data
        conflict_threshold_nm: Distance threshold for conflicts
        eps_nm: Clustering distance threshold
        min_samples: Minimum samples per cluster
        
    Returns:
        dict: Conflict zone analysis results
    """
    # Identify conflict points
    conflicts = df[df.get('conflict_flag', 0) == 1].copy()
    
    if conflicts.empty:
        return {'n_zones': 0, 'zones': pd.DataFrame()}
    
    # Cluster conflict locations
    clustered = cluster_fp_fn(conflicts, kind='conflict_flag', eps_nm=eps_nm, min_samples=min_samples)
    
    # Analyze conflict zones
    zones = clustered[clustered['cluster'] >= 0]
    
    if zones.empty:
        return {'n_zones': 0, 'zones': pd.DataFrame()}
    
    zone_stats = []
    for zone_id in zones['cluster'].unique():
        zone_data = zones[zones['cluster'] == zone_id]
        
        stats = {
            'zone_id': zone_id,
            'n_conflicts': len(zone_data),
            'center_lat': zone_data['lat_deg'].mean(),
            'center_lon': zone_data['lon_deg'].mean(),
            'radius_nm': _calculate_cluster_radius(zone_data),
            'min_separation': zone_data['min_separation_nm'].min() if 'min_separation_nm' in zone_data else np.nan,
            'avg_separation': zone_data['min_separation_nm'].mean() if 'min_separation_nm' in zone_data else np.nan,
            'agents_involved': list(zone_data['agent_id'].unique()),
            'conflict_density': len(zone_data) / max(_calculate_cluster_radius(zone_data), 0.1)
        }
        zone_stats.append(stats)
    
    return {
        'n_zones': len(zone_stats),
        'zones': pd.DataFrame(zone_stats),
        'clustered_df': clustered
    }