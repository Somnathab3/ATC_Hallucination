# viz_geographic.py
"""
Geographic visualization for ATC hallucination analysis using Folium.

Provides comprehensive mapping capabilities including:
- Baseline trajectory overlays vs. shifted trajectories
- Heat maps of conflict locations (LoS ≤ 5 NM)  
- Overlaid 5 NM safety circles around aircraft at each step
- Hallucination markers (FP/FN/TP/TN) with color/shape coding
- Time slider animation (TimestampedGeoJson / HeatMapWithTime)

Install note: folium is a pure‑Python lib that renders Leaflet maps; 
its time plugins are built‑in.
"""

import json
import math
import pandas as pd
import numpy as np
import folium
from folium import Map, Circle, PolyLine, LayerControl, FeatureGroup
from folium.plugins import HeatMap, HeatMapWithTime, TimestampedGeoJson

# Constants
NM_TO_M = 1852.0  # Nautical miles to meters conversion

def _haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in nautical miles using haversine formula."""
    R_nm = 3440.065  # Earth radius in nautical miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    return 2 * R_nm * math.asin(math.sqrt(a))

def make_basemap(df, center=None, zoom=8):
    """Create base folium map centered on data or specified location."""
    if center is None:
        clat, clon = df["lat_deg"].mean(), df["lon_deg"].mean()
    else:
        clat, clon = center
    
    m = Map(
        location=[clat, clon], 
        zoom_start=zoom, 
        tiles="OpenStreetMap", 
        control_scale=True
    )
    return m

def add_trajectories(m, df, label="baseline", color_by_agent=True):
    """Add trajectory polylines to the map, colored by agent."""
    layer = FeatureGroup(name=f"Trajectories: {label}", show=True)
    
    for aid, g in df.groupby("agent_id"):
        pts = g.sort_values("step_idx")[["lat_deg", "lon_deg"]].values.tolist()
        if len(pts) >= 2:
            color = None if not color_by_agent else _color_for_agent(aid)
            PolyLine(
                pts, 
                color=color, 
                weight=3, 
                opacity=0.8, 
                tooltip=f"{label} — {aid}"
            ).add_to(layer)
    
    layer.add_to(m)

def _color_for_agent(aid):
    """Return consistent color for each agent ID."""
    palette = {
        "A0": "#1f77b4",  # Blue
        "A1": "#ff7f0e",  # Orange  
        "A2": "#2ca02c",  # Green
        "A3": "#d62728"   # Red
    }
    return palette.get(str(aid), "#9467bd")  # Purple default

def add_safety_circles(m, df, sep_nm=5.0, every_n=5, label="5 NM safety"):
    """Add safety circles around aircraft positions."""
    layer = FeatureGroup(name=f"Safety circles ({sep_nm} NM): {label}", show=False)
    r_m = sep_nm * NM_TO_M
    
    # Sparse sampling for performance
    for (aid, g) in df.groupby("agent_id"):
        sg = g.sort_values("step_idx")
        sg = sg.iloc[::every_n, :]  # Sample every n-th step
        
        for _, row in sg.iterrows():
            Circle(
                location=[row.lat_deg, row.lon_deg],
                radius=r_m, 
                color=_color_for_agent(aid),
                fill=False, 
                weight=1, 
                opacity=0.5
            ).add_to(layer)
    
    layer.add_to(m)

def add_conflict_heatmap(m, df, label="LoS heat"):
    """Add heatmap of conflict locations (LoS points ≤ 5 NM)."""
    # Use each step position where conflict_flag==1
    conflict_pts = df.loc[df["conflict_flag"] == 1, ["lat_deg", "lon_deg"]].dropna().values.tolist()
    
    if not conflict_pts:
        return
        
    layer = FeatureGroup(name=label, show=False)
    HeatMap(
        conflict_pts, 
        radius=12, 
        blur=20, 
        min_opacity=0.3
    ).add_to(layer)
    layer.add_to(m)

def add_hallucination_markers(m, df, label="Hallucination markers"):
    """Add markers for hallucination events (FP/FN/TP/TN)."""
    # Expect fp / fn / tp / tn columns (boolean or {0,1})
    layer = FeatureGroup(name=label, show=True)
    
    icons = {
        "TP": {"color": "green", "prefix": "fa", "icon": "check"},
        "FP": {"color": "orange", "prefix": "fa", "icon": "exclamation"},
        "FN": {"color": "red", "prefix": "fa", "icon": "times"},
    }
    
    for _, r in df.iterrows():
        tag = None
        if r.get("fp", 0) == 1: 
            tag = "FP"
        elif r.get("fn", 0) == 1: 
            tag = "FN"
        elif r.get("tp", 0) == 1: 
            tag = "TP"
            
        if not tag: 
            continue
            
        folium.Marker(
            location=[r.lat_deg, r.lon_deg],
            tooltip=f"{tag} @t={int(r.sim_time_s)}s | {r.agent_id}",
            icon=folium.Icon(
                color=icons[tag]["color"], 
                icon=icons[tag]["icon"], 
                prefix=icons[tag]["prefix"]
            )
        ).add_to(layer)
    
    layer.add_to(m)

def add_timestamped_animation(m, df, label="Animated tracks"):
    """Add timestamped animation using TimestampedGeoJson."""
    # Build TimestampedGeoJson Feature per agent with LineString pieces per step
    features = []
    
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("sim_time_s")
        times = g["sim_time_s"].astype(int).tolist()
        coords = g[["lon_deg", "lat_deg"]].values.tolist()
        
        # One feature with time-indexed points
        feat = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "times": times,
                "style": {"color": _color_for_agent(aid), "weight": 3},
                "icon": {
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": _color_for_agent(aid),
                        "stroke": "true"
                    }
                }
            }
        }
        features.append(feat)
    
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT10S", 
        add_last_point=True, 
        auto_play=False, 
        loop=False,
        transition_time=200, 
        loop_button=True, 
        time_slider_drag_update=True
    ).add_to(m)

def add_time_heatmap(m, df, label="LoS heat over time"):
    """Add time-based heatmap using HeatMapWithTime."""
    # Use HeatMapWithTime where each frame = list of [lat, lon, intensity]
    g = df[df["conflict_flag"] == 1]
    if g.empty: 
        return
        
    frames = []
    for t, gt in g.groupby("step_idx"):
        frames.append(gt[["lat_deg", "lon_deg"]].values.tolist())
        
    HeatMapWithTime(
        frames, 
        radius=18, 
        auto_play=False, 
        name=label
    ).add_to(m)

def build_map(baseline_df, shifted_df=None, out_html="map.html"):
    """
    Build comprehensive map with all visualization layers.
    
    Args:
        baseline_df: DataFrame with baseline trajectory data
        shifted_df: DataFrame with shifted trajectory data (optional)
        out_html: Output HTML filename
        
    Returns:
        str: Path to saved HTML file
    """
    m = make_basemap(baseline_df)
    
    # Add trajectory layers
    add_trajectories(m, baseline_df, label="baseline")
    if shifted_df is not None:
        add_trajectories(m, shifted_df, label="shifted", color_by_agent=True)
    
    # Add analysis layers
    add_safety_circles(m, baseline_df, sep_nm=5.0, every_n=5)
    
    # Use shifted data if available, otherwise baseline
    analysis_df = shifted_df if shifted_df is not None else baseline_df
    add_conflict_heatmap(m, analysis_df)
    add_hallucination_markers(m, analysis_df)
    add_timestamped_animation(m, analysis_df)
    add_time_heatmap(m, analysis_df)
    
    # Add layer control
    LayerControl(collapsed=False).add_to(m)
    
    # Save map
    m.save(out_html)
    return out_html

def add_start_end_waypoints(m, df, waypoints=None, label="markers"):
    """Add start/end markers and waypoints to the map."""
    layer = FeatureGroup(name=f"{label}", show=True)
    
    # Start/end per agent
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("step_idx")
        if not g.empty:
            s = g.iloc[0]; e = g.iloc[-1]
            
            # Start marker
            folium.Marker(
                [s.lat_deg, s.lon_deg],
                tooltip=f"START {aid}",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(layer)
            
            # End marker - different colors based on waypoint completion
            end_color = "blue" if e.get("waypoint_reached", 0) == 1 else "red"
            end_icon = "stop" if e.get("waypoint_reached", 0) == 1 else "pause"
            
            folium.Marker(
                [e.lat_deg, e.lon_deg],
                tooltip=f"END {aid} ({'WP reached' if end_color == 'blue' else 'In progress'})",
                icon=folium.Icon(color=end_color, icon=end_icon)
            ).add_to(layer)
    
    # Waypoints if provided
    if waypoints:
        for w in waypoints:
            folium.Marker(
                [w["lat_deg"], w["lon_deg"]],
                tooltip=f"WPT {w['name']}",
                icon=folium.Icon(color="purple", icon="flag")
            ).add_to(layer)
    
    layer.add_to(m)


def create_comparison_map(baseline_df, shifted_df, map_name, out_dir, waypoints=None):
    """Create a comprehensive comparison map with baseline and shifted trajectories."""
    m = make_basemap(baseline_df)
    
    # Add trajectory layers
    add_trajectories(m, baseline_df, label="baseline")
    if shifted_df is not None:
        add_trajectories(m, shifted_df, label="shifted")
    
    # Add safety and analysis layers using shifted data if available
    analysis_df = shifted_df if shifted_df is not None else baseline_df
    add_safety_circles(m, analysis_df, sep_nm=5.0, every_n=5)
    add_conflict_heatmap(m, analysis_df)
    add_hallucination_markers(m, analysis_df)
    add_time_heatmap(m, analysis_df)
    
    # Add start/end/waypoint markers
    add_start_end_waypoints(m, baseline_df, waypoints=waypoints)
    
    # Add layer control
    LayerControl(collapsed=False).add_to(m)
    
    # Save map
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{map_name}.html")
    m.save(out_path)
    return out_path


def add_baseline_overlay(m, baseline_df, shifted_df, label="baseline vs shifted"):
    """Add baseline trajectory overlay for comparison with shifted trajectories."""
    if baseline_df is None or baseline_df.empty:
        return
        
    # Baseline trajectories with dashed lines
    baseline_layer = FeatureGroup(name=f"Baseline: {label}", show=True)
    
    for aid, g in baseline_df.groupby("agent_id"):
        pts = g.sort_values("step_idx")[["lat_deg", "lon_deg"]].values.tolist()
        if len(pts) >= 2:
            PolyLine(
                pts, 
                color=_color_for_agent(aid), 
                weight=2, 
                opacity=0.6,
                dashArray="10, 10",  # Dashed line for baseline
                tooltip=f"Baseline — {aid}"
            ).add_to(baseline_layer)
    
    # Shifted trajectories with solid lines
    shifted_layer = FeatureGroup(name=f"Shifted: {label}", show=True)
    
    for aid, g in shifted_df.groupby("agent_id"):
        pts = g.sort_values("step_idx")[["lat_deg", "lon_deg"]].values.tolist()
        if len(pts) >= 2:
            PolyLine(
                pts, 
                color=_color_for_agent(aid), 
                weight=3, 
                opacity=0.8,
                tooltip=f"Shifted — {aid}"
            ).add_to(shifted_layer)
    
    baseline_layer.add_to(m)
    shifted_layer.add_to(m)


def create_enhanced_comparison_map(baseline_df, shifted_df, shift_info, out_dir):
    """
    Create an enhanced comparison map with baseline overlay and detailed shift information.
    
    Args:
        baseline_df: DataFrame with baseline trajectory data
        shifted_df: DataFrame with shifted trajectory data
        shift_info: Dict with shift metadata (test_id, description, etc.)
        out_dir: Output directory for the map
        
    Returns:
        str: Path to saved HTML file
    """
    import os
    
    # Create base map
    m = make_basemap(shifted_df if shifted_df is not None else baseline_df)
    
    # Add baseline overlay
    if baseline_df is not None and not baseline_df.empty:
        add_baseline_overlay(m, baseline_df, shifted_df, shift_info.get("description", "comparison"))
    
    # Add analysis layers for shifted data
    if shifted_df is not None and not shifted_df.empty:
        add_safety_circles(m, shifted_df, sep_nm=5.0, every_n=3)
        add_conflict_heatmap(m, shifted_df, label="Shift conflict zones")
        add_hallucination_markers(m, shifted_df, label="Shift hallucinations")
        add_time_heatmap(m, shifted_df, label="Shift LoS over time")
    
    # Add start/end markers for both datasets
    if baseline_df is not None and not baseline_df.empty:
        add_start_end_waypoints(m, baseline_df, label="Baseline markers")
    if shifted_df is not None and not shifted_df.empty:
        add_start_end_waypoints(m, shifted_df, label="Shifted markers")
    
    # Add layer control
    LayerControl(collapsed=False).add_to(m)
    
    # Add title to map using a different approach
    title_html = f'''
    <h3 align="center" style="font-size:20px"><b>{shift_info.get("description", "Trajectory Comparison")}</b></h3>
    <p align="center">Test ID: {shift_info.get("test_id", "unknown")}<br>
    Shift Type: {shift_info.get("shift_type", "unknown")} | 
    Shift Value: {shift_info.get("shift_value", "N/A")}<br>
    Target Agent: {shift_info.get("target_agent", "unknown")}</p>
    '''
    
    # Add title as HTML element
    try:
        from folium import Element
        title_element = Element(title_html)
        m.get_root().add_child(title_element)
    except Exception:
        # Fallback: just save without title
        pass
    
    # Save map
    os.makedirs(out_dir, exist_ok=True)
    map_filename = f"comparison_{shift_info.get('test_id', 'unknown')}.html"
    out_path = os.path.join(out_dir, map_filename)
    m.save(out_path)
    
    return out_path

def create_baseline_vs_shifted_comparison(results_dir: str, scenario_name: str, out_dir: str = "trajectory_maps"):
    """
    Create comparative maps showing baseline vs shifted trajectories.
    
    Args:
        results_dir: Directory containing shift test results
        scenario_name: Name of the scenario
        out_dir: Output directory for maps
        
    Returns:
        List of paths to generated map files
    """
    import os
    import glob
    import pandas as pd
    
    os.makedirs(os.path.join(results_dir, out_dir), exist_ok=True)
    
    # Find all shift directories
    shifts_dir = os.path.join(results_dir, "shifts")
    if not os.path.exists(shifts_dir):
        print(f"Warning: No shifts directory found in {results_dir}")
        return []
    
    generated_maps = []
    
    # Process each shift type
    shift_dirs = [d for d in os.listdir(shifts_dir) if os.path.isdir(os.path.join(shifts_dir, d))]
    
    for shift_dir_name in shift_dirs[:5]:  # Limit to first 5 for demo
        shift_path = os.path.join(shifts_dir, shift_dir_name)
        
        # Find CSV files in this shift directory
        # Look for both old pattern (traj_ep_*.csv) and new pattern (traj_*_ep*.csv)
        csv_files = glob.glob(os.path.join(shift_path, "traj_ep_*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(shift_path, "traj_*_ep*.csv"))
        
        if not csv_files:
            print(f"Warning: No trajectory CSV files found in {shift_path}")
            continue
            
        # Load trajectory data
        all_trajectories = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['episode_file'] = os.path.basename(csv_file)
                all_trajectories.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue
        
        if not all_trajectories:
            continue
            
        # Combine all episodes for this shift
        combined_df = pd.concat(all_trajectories, ignore_index=True)
        
        # Create the comparison map
        try:
            map_html = os.path.join(results_dir, out_dir, f"trajectory_comparison_{shift_dir_name}.html")
            _create_trajectory_comparison_map(combined_df, shift_dir_name, map_html)
            generated_maps.append(map_html)
            print(f"Generated trajectory comparison map: {map_html}")
        except Exception as e:
            print(f"Warning: Failed to create map for {shift_dir_name}: {e}")
    
    return generated_maps


def _create_trajectory_comparison_map(df: pd.DataFrame, shift_name: str, out_html: str):
    """
    Create a single map comparing trajectories for a specific shift.
    
    Args:
        df: Combined trajectory DataFrame for all episodes of this shift
        shift_name: Name of the shift being analyzed
        out_html: Output HTML file path
    """
    # Calculate map center
    center_lat = df['lat_deg'].mean()
    center_lon = df['lon_deg'].mean()
    
    # Create base map
    m = Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
    
    # Add trajectory layers for each episode and agent
    colors = {'A1': '#1f77b4', 'A2': '#ff7f0e', 'A3': '#2ca02c'}
    
    # Group by episode and agent
    for episode_file, episode_df in df.groupby('episode_file'):
        episode_num = str(episode_file).split('_')[-1].split('.')[0]  # Extract episode number
        
        for agent_id, agent_df in episode_df.groupby('agent_id'):
            # Sort by step_idx to ensure proper trajectory order
            agent_df = agent_df.sort_values('step_idx')
            
            # Create trajectory line
            trajectory_points = agent_df[['lat_deg', 'lon_deg']].values.tolist()
            
            if len(trajectory_points) >= 2:
                # Different line styles for different episodes
                line_style = {'weight': 3, 'opacity': 0.8}
                if episode_num == '0001':
                    line_style['dashArray'] = '5, 5'  # Dashed for episode 1
                
                PolyLine(
                    trajectory_points,
                    color=colors.get(str(agent_id), '#9467bd'),
                    tooltip=f"{shift_name} - {agent_id} - Episode {episode_num}",
                    **line_style
                ).add_to(m)
                
                # Add start and end markers
                start_point = trajectory_points[0]
                end_point = trajectory_points[-1]
                
                # Start marker (green)
                folium.CircleMarker(
                    start_point,
                    radius=6,
                    color='green',
                    fill=True,
                    popup=f"{agent_id} Start - Ep {episode_num}",
                    tooltip=f"{agent_id} Start"
                ).add_to(m)
                
                # End marker (red if not reached waypoint, blue if reached)
                last_row = agent_df.iloc[-1]
                end_color = 'blue' if last_row.get('waypoint_reached', 0) == 1 else 'red'
                
                folium.CircleMarker(
                    end_point,
                    radius=6,
                    color=end_color,
                    fill=True,
                    popup=f"{agent_id} End - Ep {episode_num} ({'Waypoint Reached' if end_color == 'blue' else 'In Progress'})",
                    tooltip=f"{agent_id} End"
                ).add_to(m)
    
    # Add conflict zones (where min_separation < 5 NM)
    conflict_points = df[df['min_separation_nm'] < 5.0]
    if not conflict_points.empty:
        conflict_layer = FeatureGroup(name="Conflict Zones", show=True)
        
        for _, row in conflict_points.iterrows():
            folium.CircleMarker(
                [row['lat_deg'], row['lon_deg']],
                radius=8,
                color='red',
                fill=True,
                fillOpacity=0.6,
                popup=f"Conflict: {row['min_separation_nm']:.2f} NM @ t={row['sim_time_s']:.0f}s",
                tooltip="Conflict Zone"
            ).add_to(conflict_layer)
        
        conflict_layer.add_to(m)
    
    # Add layer control
    LayerControl(collapsed=False).add_to(m)
    
    # Save map
    m.save(out_html)