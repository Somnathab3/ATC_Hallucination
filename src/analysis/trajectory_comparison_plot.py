"""
Plotly-based trajectory comparison visualization for shift analysis.

This module creates interactive scatter plots showing trajectory patterns without geographic maps,
making it easier to understand how parameter shifts affect aircraft movement patterns.

Key features:
- Shows trajectory deviations from baseline for each shift type
- Color-codes shifts by type (speed, position, heading) and agent
- Interactive plots with hover information and zoom capabilities
- Batch processing for comparing multiple shift configurations
- Statistical analysis of trajectory spread for each shift type
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import json


def _read_trajectory_data(csv_path: str) -> pd.DataFrame:
    """Read trajectory CSV and prepare it for analysis."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    required_cols = ['step_idx', 'agent_id', 'lat_deg', 'lon_deg']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")
    
    # Sort for proper trajectory ordering
    df = df.sort_values(['agent_id', 'step_idx'])
    return df


def _normalize_trajectories(baseline_df: pd.DataFrame, shift_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize trajectories relative to baseline starting positions.
    This makes it easier to see trajectory deviations.
    """
    # Get baseline starting positions for each agent
    baseline_starts = {}
    for agent_id in baseline_df['agent_id'].unique():
        agent_data = baseline_df[baseline_df['agent_id'] == agent_id].iloc[0]
        baseline_starts[agent_id] = {
            'lat': agent_data['lat_deg'],
            'lon': agent_data['lon_deg']
        }
    
    # Normalize both datasets
    def normalize_df(df, starts):
        df_norm = df.copy()
        for agent_id in df['agent_id'].unique():
            if agent_id in starts:
                mask = df_norm['agent_id'] == agent_id
                df_norm.loc[mask, 'lat_normalized'] = df_norm.loc[mask, 'lat_deg'] - starts[agent_id]['lat']
                df_norm.loc[mask, 'lon_normalized'] = df_norm.loc[mask, 'lon_deg'] - starts[agent_id]['lon']
            else:
                # If agent not in baseline (shouldn't happen), use agent's own start
                agent_data = df[df['agent_id'] == agent_id].iloc[0]
                mask = df_norm['agent_id'] == agent_id
                df_norm.loc[mask, 'lat_normalized'] = df_norm.loc[mask, 'lat_deg'] - agent_data['lat_deg']
                df_norm.loc[mask, 'lon_normalized'] = df_norm.loc[mask, 'lon_deg'] - agent_data['lon_deg']
        
        return df_norm
    
    baseline_norm = normalize_df(baseline_df, baseline_starts)
    shift_norm = normalize_df(shift_df, baseline_starts)
    
    return baseline_norm, shift_norm


def _extract_shift_metadata(shift_name: str) -> Dict[str, str]:
    """Extract shift type, agent, and magnitude from shift name."""
    # Parse shift names like "speed_micro_A1_+5kt", "aircraft_A1_B737", "waypoint_micro_A2_north_0.05deg"
    parts = shift_name.split('_')
    
    if len(parts) < 3:
        return {'type': 'unknown', 'agent': 'unknown', 'magnitude': 'unknown', 'range': 'unknown'}
    
    shift_type = parts[0]
    if shift_type == 'pos':
        shift_type = f"{parts[0]}_{parts[1]}"  # pos_closer or pos_lateral
        range_type = parts[2] if len(parts) > 2 else 'unknown'
        agent = parts[3] if len(parts) > 3 else 'unknown'
        magnitude = '_'.join(parts[4:]) if len(parts) > 4 else 'unknown'
    elif shift_type == 'hdg':
        range_type = parts[1] if len(parts) > 1 else 'unknown'
        agent = parts[2] if len(parts) > 2 else 'unknown'
        magnitude = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
    elif shift_type == 'aircraft':
        range_type = 'type'  # Special case for aircraft type
        agent = parts[1] if len(parts) > 1 else 'unknown'
        magnitude = parts[2] if len(parts) > 2 else 'unknown'
    elif shift_type == 'waypoint':
        range_type = parts[1] if len(parts) > 1 else 'unknown'
        agent = parts[2] if len(parts) > 2 else 'unknown'
        magnitude = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
    else:  # speed
        range_type = parts[1] if len(parts) > 1 else 'unknown'
        agent = parts[2] if len(parts) > 2 else 'unknown'
        magnitude = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
    
    return {
        'type': shift_type,
        'agent': agent,
        'magnitude': magnitude,
        'range': range_type
    }


def _read_scenario_waypoints(scenario_path: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """Read waypoint information from scenario file if available."""
    waypoints = {}
    
    if scenario_path and os.path.exists(scenario_path):
        try:
            import json
            with open(scenario_path, 'r') as f:
                scenario_data = json.load(f)
            
            for agent in scenario_data.get('agents', []):
                agent_id = agent.get('id')
                waypoint = agent.get('waypoint', {})
                if agent_id and waypoint and 'lat' in waypoint and 'lon' in waypoint:
                    waypoints[agent_id] = (waypoint['lat'], waypoint['lon'])
        except Exception as e:
            print(f"Warning: Could not read waypoints from scenario: {e}")
    
    return waypoints


def create_trajectory_comparison_plot(baseline_csv: str, shift_csvs: Dict[str, str], 
                                    out_html: str = "trajectory_comparison.html",
                                    title: Optional[str] = None,
                                    scenario_path: Optional[str] = None) -> bool:
    """
    Create a Plotly-based trajectory comparison plot.
    
    Args:
        baseline_csv: Path to baseline trajectory CSV
        shift_csvs: Dictionary of {shift_name: csv_path} for shift trajectories
        out_html: Output HTML file path
        title: Optional custom title
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read baseline data
        baseline_df = _read_trajectory_data(baseline_csv)
        
        # Read waypoint information
        waypoints = _read_scenario_waypoints(scenario_path)
        
        # Color schemes for different shift types and agents
        shift_type_colors = {
            'speed': ['#FF6B6B', '#FF8E8E', '#FFB1B1'],      # Red family
            'pos_closer': ['#4ECDC4', '#7DD3DB', '#ADE9E6'], # Teal family  
            'pos_lateral': ['#45B7D1', '#6BC7E1', '#91D7F1'], # Blue family
            'hdg': ['#96CEB4', '#B5D6C7', '#D4E3DA'],        # Green family
            'heading': ['#96CEB4', '#B5D6C7', '#D4E3DA'],    # Green family (alt name)
            'aircraft': ['#9B59B6', '#BB8FCE', '#D7BDE2'],   # Purple family
            'waypoint': ['#F39C12', '#F8C471', '#FCF3CF']    # Orange/Yellow family
        }
        
        agent_colors = {
            'A1': ['#1f77b4', '#aec7e8'],  # Blue family
            'A2': ['#ff7f0e', '#ffbb78'],  # Orange family  
            'A3': ['#2ca02c', '#98df8a']   # Green family
        }
        
        # Create subplots: one for each agent
        agents = sorted(baseline_df['agent_id'].unique())
        fig = make_subplots(
            rows=1, cols=len(agents),
            subplot_titles=[f"Agent {agent}" for agent in agents],
            horizontal_spacing=0.1
        )
        
        # Plot baseline trajectories
        baseline_norm, _ = _normalize_trajectories(baseline_df, baseline_df)
        
        for i, agent in enumerate(agents, 1):
            agent_data = baseline_norm[baseline_norm['agent_id'] == agent]
            
            fig.add_trace(
                go.Scatter(
                    x=agent_data['lon_normalized'],
                    y=agent_data['lat_normalized'],
                    mode='lines+markers',
                    name=f'Baseline {agent}',
                    line=dict(color='black', width=3),
                    marker=dict(size=4, color='black', symbol='circle'),
                    hovertemplate=f"<b>Baseline {agent}</b><br>" +
                                "Step: %{customdata[0]}<br>" +
                                "Δ Longitude: %{x:.6f}°<br>" +
                                "Δ Latitude: %{y:.6f}°<br>" +
                                "<extra></extra>",
                    customdata=agent_data[['step_idx']].values,
                    showlegend=(i == 1)  # Only show legend for first agent
                ),
                row=1, col=i
            )
            
            # Add red star for start point (baseline)
            if len(agent_data) > 0:
                start_point = agent_data.iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[start_point['lon_normalized']],
                        y=[start_point['lat_normalized']],
                        mode='markers',
                        name='Start Point' if i == 1 else None,  # Only add to legend once
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='star',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate=f"<b>Start Point - {agent}</b><br>" +
                                    "Δ Longitude: %{x:.6f}°<br>" +
                                    "Δ Latitude: %{y:.6f}°<br>" +
                                    "<extra></extra>",
                        showlegend=(i == 1)  # Only show in legend for first agent
                    ),
                    row=1, col=i
                )
        
        # Plot shift trajectories
        for shift_name, shift_csv in shift_csvs.items():
            try:
                shift_df = _read_trajectory_data(shift_csv)
                _, shift_norm = _normalize_trajectories(baseline_df, shift_df)
                
                # Extract shift metadata
                metadata = _extract_shift_metadata(shift_name)
                shift_type = metadata['type']
                target_agent = metadata['agent']
                
                # Get colors for this shift type
                type_colors = shift_type_colors.get(shift_type, ['#666666', '#888888', '#AAAAAA'])
                
                for i, agent in enumerate(agents, 1):
                    agent_data = shift_norm[shift_norm['agent_id'] == agent]
                    
                    if len(agent_data) == 0:
                        continue
                    
                    # Different styling for target agent vs others
                    if agent == target_agent:
                        # Target agent: solid line, primary color
                        line_style = dict(color=type_colors[0], width=2, dash='solid')
                        marker_style = dict(size=3, color=type_colors[0], symbol='diamond')
                        opacity = 0.8
                    else:
                        # Other agents: dashed line, secondary color
                        line_style = dict(color=type_colors[1], width=1, dash='dash')
                        marker_style = dict(size=2, color=type_colors[1], symbol='circle')
                        opacity = 0.5
                    
                    fig.add_trace(
                        go.Scatter(
                            x=agent_data['lon_normalized'],
                            y=agent_data['lat_normalized'],
                            mode='lines+markers',
                            name=f'{shift_type} {agent}' if agent == target_agent else f'{shift_type} {agent} (indirect)',
                            line=line_style,
                            marker=marker_style,
                            opacity=opacity,
                            hovertemplate=f"<b>{shift_name}</b><br>" +
                                        f"Agent: {agent}<br>" +
                                        f"Type: {shift_type}<br>" +
                                        f"Target: {target_agent}<br>" +
                                        "Step: %{customdata[0]}<br>" +
                                        "Δ Longitude: %{x:.6f}°<br>" +
                                        "Δ Latitude: %{y:.6f}°<br>" +
                                        "<extra></extra>",
                            customdata=agent_data[['step_idx']].values,
                            showlegend=(i == len(agents) and agent == target_agent)  # Only show legend for target agent on last subplot
                        ),
                        row=1, col=i
                    )
                    
                    # Add red star for start point (shift trajectories)
                    if len(agent_data) > 0:
                        start_point = agent_data.iloc[0]
                        fig.add_trace(
                            go.Scatter(
                                x=[start_point['lon_normalized']],
                                y=[start_point['lat_normalized']],
                                mode='markers',
                                name=None,  # Don't add to legend (already added for baseline)
                                marker=dict(
                                    size=12,
                                    color='red',
                                    symbol='star',
                                    line=dict(width=2, color='darkred')
                                ),
                                hovertemplate=f"<b>Start Point - {agent} ({shift_name})</b><br>" +
                                            "Δ Longitude: %{x:.6f}°<br>" +
                                            "Δ Latitude: %{y:.6f}°<br>" +
                                            "<extra></extra>",
                                showlegend=False  # Don't show in legend
                            ),
                            row=1, col=i
                        )
                    
            except Exception as e:
                print(f"Warning: Failed to process {shift_name}: {e}")
                continue
        
        # Add waypoint markers if available
        if waypoints:
            baseline_norm_sample = baseline_norm.iloc[0] if len(baseline_norm) > 0 else None
            if baseline_norm_sample is not None:
                for i, agent in enumerate(agents, 1):
                    if agent in waypoints:
                        # Get baseline starting position to normalize waypoint
                        agent_baseline_start = baseline_df[baseline_df['agent_id'] == agent].iloc[0]
                        start_lat = agent_baseline_start['lat_deg']
                        start_lon = agent_baseline_start['lon_deg']
                        
                        # Normalize waypoint relative to baseline start
                        wp_lat, wp_lon = waypoints[agent]
                        wp_lat_norm = wp_lat - start_lat
                        wp_lon_norm = wp_lon - start_lon
                        
                        # Add waypoint marker
                        fig.add_trace(
                            go.Scatter(
                                x=[wp_lon_norm],
                                y=[wp_lat_norm],
                                mode='markers',
                                name=f'Waypoint {agent}',
                                marker=dict(
                                    size=12,
                                    color='gold',
                                    symbol='star',
                                    line=dict(width=2, color='black')
                                ),
                                hovertemplate=f"<b>Waypoint {agent}</b><br>" +
                                            "Δ Longitude: %{x:.6f}°<br>" +
                                            "Δ Latitude: %{y:.6f}°<br>" +
                                            f"Absolute: {wp_lat:.4f}°, {wp_lon:.4f}°<br>" +
                                            "<extra></extra>",
                                showlegend=(i == len(agents))  # Only show legend for last agent
                            ),
                            row=1, col=i
                        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title or "Trajectory Comparison: Baseline vs Parameter Shifts",
                x=0.5,
                font=dict(size=16)
            ),
            height=600,
            width=300 * len(agents) + 200,  # Add extra width for right-side legend
            hovermode='closest',
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="top",
                y=1.0,
                xanchor="left", 
                x=1.02  # Position to the right of the plot area
            )
        )
        
        # Update axes labels
        for i in range(len(agents)):
            fig.update_xaxes(
                title_text="Δ Longitude (degrees)" if i == len(agents)//2 else "",
                row=1, col=i+1
            )
            fig.update_yaxes(
                title_text="Δ Latitude (degrees)" if i == 0 else "",
                row=1, col=i+1
            )
        
        # Add annotations for shift types
        shift_types_legend = {}
        for shift_name in shift_csvs.keys():
            metadata = _extract_shift_metadata(shift_name)
            shift_type = metadata['type']
            if shift_type not in shift_types_legend:
                shift_types_legend[shift_type] = shift_type_colors.get(shift_type, ['#666666'])[0]
        
        legend_text = "Shift Types: " + " | ".join([
            f"<span style='color:{color}'>{stype}</span>" 
            for stype, color in shift_types_legend.items()
        ])
        
        fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            showarrow=False,
            font=dict(size=12)
        )
        
        # Save the plot
        os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
        fig.write_html(out_html)
        
        print(f"Saved trajectory comparison plot: {out_html}")
        return True
        
    except Exception as e:
        print(f"Error creating trajectory comparison plot: {e}")
        return False


def create_multi_agent_combined_plot(baseline_csv: str, shift_csvs: Dict[str, str], 
                                    output_file: str, title: str = "Multi-Agent Trajectory Analysis",
                                    scenario_path: Optional[str] = None) -> bool:
    """
    Create a single plot showing multiple agents with different shift types simultaneously.
    
    Args:
        baseline_csv: Path to baseline trajectory CSV
        shift_csvs: Dictionary mapping shift names to CSV file paths
        output_file: Output HTML file path
        title: Plot title
        scenario_path: Optional path to scenario file for waypoint markers
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read baseline data
        baseline_df = _read_trajectory_data(baseline_csv)
        agents = sorted(baseline_df['agent_id'].unique())
        
        # Create single plot figure
        fig = go.Figure()
        
        # Define enhanced color schemes for each shift type
        shift_type_colors = {
            'speed': ['#FF4444', '#FF6666', '#FF8888'],      # Red tones
            'pos_closer': ['#4444FF', '#6666FF', '#8888FF'], # Blue tones  
            'pos_lateral': ['#44AAFF', '#66BBFF', '#88CCFF'], # Light blue tones
            'hdg': ['#44FF44', '#66FF66', '#88FF88'],         # Green tones
            'aircraft': ['#FFAA00', '#FFBB22', '#FFCC44'],   # Orange tones
            'waypoint': ['#AA44FF', '#BB66FF', '#CC88FF']    # Purple tones
        }
        
        # Read waypoints if scenario provided
        waypoints = _read_scenario_waypoints(scenario_path)
        
        # Plot baseline trajectories for all agents
        for i, agent in enumerate(agents):
            agent_data = baseline_df[baseline_df['agent_id'] == agent]
            if len(agent_data) == 0:
                continue
                
            # Normalize baseline for this agent
            baseline_norm, _ = _normalize_trajectories(agent_data, agent_data)
            
            fig.add_trace(
                go.Scatter(
                    x=baseline_norm['lon_normalized'],
                    y=baseline_norm['lat_normalized'],
                    mode='lines+markers',
                    name=f'Baseline {agent}',
                    line=dict(color='black', width=2, dash='solid'),
                    marker=dict(size=4, color='black', symbol='circle'),
                    hovertemplate=f"<b>Baseline {agent}</b><br>" +
                                "Step: %{customdata[0]}<br>" +
                                "Δ Longitude: %{x:.6f}°<br>" +
                                "Δ Latitude: %{y:.6f}°<br>" +
                                "<extra></extra>",
                    customdata=baseline_norm[['step_idx']].values,
                    legendgroup=f"baseline_{agent}",
                    showlegend=True
                )
            )
            
            # Add waypoint markers for baseline
            if agent in waypoints:
                wpt_lat, wpt_lon = waypoints[agent]
                # Convert waypoint to normalized coordinates (approximation)
                start_lat = baseline_norm['lat_normalized'].iloc[0] + agent_data['lat_deg'].iloc[0]
                start_lon = baseline_norm['lon_normalized'].iloc[0] + agent_data['lon_deg'].iloc[0]
                norm_wpt_lat = wpt_lat - start_lat
                norm_wpt_lon = wpt_lon - start_lon
                
                fig.add_trace(
                    go.Scatter(
                        x=[norm_wpt_lon],
                        y=[norm_wpt_lat],
                        mode='markers',
                        name=f'Waypoint {agent}',
                        marker=dict(size=12, color='black', symbol='star', 
                                  line=dict(color='white', width=2)),
                        hovertemplate=f"<b>Waypoint {agent}</b><br>" +
                                    f"Lat: {wpt_lat:.4f}°<br>" +
                                    f"Lon: {wpt_lon:.4f}°<br>" +
                                    "<extra></extra>",
                        legendgroup=f"baseline_{agent}",
                        showlegend=False
                    )
                )
        
        # Plot shift trajectories with different colors for each shift type
        for shift_name, shift_csv in shift_csvs.items():
            try:
                shift_df = _read_trajectory_data(shift_csv)
                _, shift_norm = _normalize_trajectories(baseline_df, shift_df)
                
                # Extract shift metadata
                metadata = _extract_shift_metadata(shift_name)
                shift_type = metadata['type']
                target_agent = metadata['agent']
                
                # Get colors for this shift type
                type_colors = shift_type_colors.get(shift_type, ['#666666', '#888888', '#AAAAAA'])
                
                for agent in agents:
                    agent_data = shift_norm[shift_norm['agent_id'] == agent]
                    
                    if len(agent_data) == 0:
                        continue
                    
                    # Different styling for target agent vs others
                    if agent == target_agent:
                        # Target agent: solid line, primary color, more prominent
                        line_style = dict(color=type_colors[0], width=3, dash='solid')
                        marker_style = dict(size=5, color=type_colors[0], symbol='diamond')
                        opacity = 0.9
                        legend_suffix = ""
                    else:
                        # Other agents: dashed line, secondary color, less prominent
                        line_style = dict(color=type_colors[1], width=2, dash='dash')
                        marker_style = dict(size=3, color=type_colors[1], symbol='circle')
                        opacity = 0.6
                        legend_suffix = " (indirect)"
                    
                    # Create display name for shift
                    display_name = f'{shift_type.title()} {agent}{legend_suffix}'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=agent_data['lon_normalized'],
                            y=agent_data['lat_normalized'],
                            mode='lines+markers',
                            name=display_name,
                            line=line_style,
                            marker=marker_style,
                            opacity=opacity,
                            hovertemplate=f"<b>{display_name}</b><br>" +
                                        f"Shift: {shift_name}<br>" +
                                        "Step: %{customdata[0]}<br>" +
                                        "Δ Longitude: %{x:.6f}°<br>" +
                                        "Δ Latitude: %{y:.6f}°<br>" +
                                        "<extra></extra>",
                            customdata=agent_data[['step_idx']].values,
                            legendgroup=f"{shift_type}_{agent}",
                            showlegend=True
                        )
                    )
                    
            except Exception as e:
                print(f"Error processing shift {shift_name}: {e}")
                continue
        
        # Update layout with proper legend positioning
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title="Δ Longitude (degrees)",
            yaxis_title="Δ Latitude (degrees)",
            height=700,
            width=1200,  # Wider to accommodate legend
            hovermode='closest',
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,  # Position to the right of the plot area
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Add subtitle annotation
        num_shifts = len(shift_csvs)
        num_agents = len(agents)
        subtitle = f"Combined view of {num_agents} agents across {num_shifts} parameter shifts"
        
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        # Save plot
        fig.write_html(output_file)
        print(f"Saved multi-agent combined plot: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating multi-agent combined plot: {e}")
        return False


def create_shift_analysis_dashboard(baseline_dir: str, shifts_dir: str, 
                                  viz_dir: str, scenario_path: Optional[str] = None) -> str:
    """
    Create a comprehensive dashboard analyzing trajectory shifts by type.
    
    Args:
        baseline_dir: Directory containing baseline trajectory CSV
        shifts_dir: Directory containing shift subdirectories
        viz_dir: Output directory for visualizations
        
    Returns:
        Path to the generated dashboard HTML file
    """
    # Find baseline CSV
    baseline_csvs = [f for f in os.listdir(baseline_dir) 
                    if f.startswith('traj_') and f.endswith('.csv')]
    
    if not baseline_csvs:
        raise FileNotFoundError(f"No baseline trajectory CSV found in {baseline_dir}")
    
    baseline_csv = os.path.join(baseline_dir, baseline_csvs[0])
    baseline_df = _read_trajectory_data(baseline_csv)
    
    # Collect all shift CSVs
    shift_data = {}
    for shift_name in os.listdir(shifts_dir):
        if shift_name == "baseline":
            continue
            
        shift_path = os.path.join(shifts_dir, shift_name)
        if not os.path.isdir(shift_path):
            continue
            
        shift_csvs = [f for f in os.listdir(shift_path)
                     if f.startswith('traj_') and f.endswith('.csv')]
        
        if shift_csvs:
            shift_data[shift_name] = os.path.join(shift_path, shift_csvs[0])
    
    if not shift_data:
        raise ValueError("No shift trajectory data found")
    
    # Group shifts by type for better visualization
    shifts_by_type = {}
    for shift_name, csv_path in shift_data.items():
        metadata = _extract_shift_metadata(shift_name)
        shift_type = metadata['type']
        
        if shift_type not in shifts_by_type:
            shifts_by_type[shift_type] = {}
        shifts_by_type[shift_type][shift_name] = csv_path
    
    # Create individual plots for each shift type
    generated_plots = []
    
    for shift_type, type_shifts in shifts_by_type.items():
        output_file = os.path.join(viz_dir, f"trajectory_analysis_{shift_type}.html")
        
        success = create_trajectory_comparison_plot(
            baseline_csv=baseline_csv,
            shift_csvs=type_shifts,
            out_html=output_file,
            title=f"Trajectory Analysis: {shift_type.replace('_', ' ').title()} Shifts",
            scenario_path=scenario_path
        )
        
        if success:
            generated_plots.append(output_file)
            print(f"  Generated {shift_type} analysis: trajectory_analysis_{shift_type}.html")
    
    # Create overall comparison with all shifts
    all_shifts_file = os.path.join(viz_dir, "trajectory_analysis_all_shifts.html")
    success = create_trajectory_comparison_plot(
        baseline_csv=baseline_csv,
        shift_csvs=shift_data,
        out_html=all_shifts_file,
        title="Trajectory Analysis: All Parameter Shifts vs Baseline",
        scenario_path=scenario_path
    )
    
    if success:
        generated_plots.append(all_shifts_file)
        print(f"  Generated combined analysis: trajectory_analysis_all_shifts.html")
    
    # Create multi-agent combined plot showing all agents with different shifts in one graph
    multi_agent_file = os.path.join(viz_dir, "trajectory_analysis_multi_agent_combined.html")
    success = create_multi_agent_combined_plot(
        baseline_csv=baseline_csv,
        shift_csvs=shift_data,
        output_file=multi_agent_file,
        title="Multi-Agent Combined Analysis: All Agents & Shifts",
        scenario_path=scenario_path
    )
    
    if success:
        generated_plots.append(multi_agent_file)
        print(f"  Generated multi-agent combined analysis: trajectory_analysis_multi_agent_combined.html")
    
    # Create summary statistics
    _create_trajectory_statistics(baseline_df, shift_data, viz_dir)
    
    # Create dashboard index
    dashboard_file = _create_trajectory_dashboard_index(viz_dir, generated_plots, shifts_by_type)
    
    print(f"\nGenerated trajectory analysis dashboard: {dashboard_file}")
    return dashboard_file


def _create_trajectory_statistics(baseline_df: pd.DataFrame, shift_data: Dict[str, str], viz_dir: str):
    """Create statistical analysis of trajectory deviations."""
    
    stats = {
        'baseline_stats': {},
        'shift_stats': {},
        'deviations': {}
    }
    
    # Baseline statistics
    for agent in baseline_df['agent_id'].unique():
        agent_data = baseline_df[baseline_df['agent_id'] == agent]
        
        # Calculate path length
        path_length = 0.0
        for i in range(1, len(agent_data)):
            lat1, lon1 = agent_data.iloc[i-1][['lat_deg', 'lon_deg']]
            lat2, lon2 = agent_data.iloc[i][['lat_deg', 'lon_deg']]
            # Simple distance approximation
            path_length += np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111.32  # Convert to km
        
        stats['baseline_stats'][agent] = {
            'path_length_km': path_length,
            'lat_range': float(agent_data['lat_deg'].max() - agent_data['lat_deg'].min()),
            'lon_range': float(agent_data['lon_deg'].max() - agent_data['lon_deg'].min()),
            'steps': len(agent_data)
        }
    
    # Shift statistics and deviations
    for shift_name, csv_path in shift_data.items():
        try:
            shift_df = _read_trajectory_data(csv_path)
            metadata = _extract_shift_metadata(shift_name)
            
            stats['shift_stats'][shift_name] = {
                'metadata': metadata,
                'agents': {}
            }
            
            # Calculate deviations from baseline
            baseline_norm, shift_norm = _normalize_trajectories(baseline_df, shift_df)
            
            max_deviation = 0.0
            avg_deviation = 0.0
            
            for agent in shift_df['agent_id'].unique():
                agent_shift = shift_norm[shift_norm['agent_id'] == agent]
                agent_baseline = baseline_norm[baseline_norm['agent_id'] == agent]
                
                if len(agent_shift) == 0 or len(agent_baseline) == 0:
                    continue
                
                # Calculate deviations at each step
                deviations = []
                min_steps = min(len(agent_shift), len(agent_baseline))
                
                for i in range(min_steps):
                    shift_point = agent_shift.iloc[i]
                    baseline_point = agent_baseline.iloc[i]
                    
                    deviation = np.sqrt(
                        (shift_point['lat_normalized'] - baseline_point['lat_normalized'])**2 +
                        (shift_point['lon_normalized'] - baseline_point['lon_normalized'])**2
                    ) * 111.32  # Convert to km
                    
                    deviations.append(deviation)
                
                if deviations:
                    agent_max_dev = max(deviations)
                    agent_avg_dev = np.mean(deviations)
                    
                    stats['shift_stats'][shift_name]['agents'][agent] = {
                        'max_deviation_km': agent_max_dev,
                        'avg_deviation_km': agent_avg_dev,
                        'deviations': deviations
                    }
                    
                    max_deviation = max(max_deviation, agent_max_dev)
                    avg_deviation = max(avg_deviation, agent_avg_dev)
            
            stats['deviations'][shift_name] = {
                'max_deviation_km': max_deviation,
                'avg_deviation_km': avg_deviation
            }
            
        except Exception as e:
            print(f"Warning: Failed to analyze {shift_name}: {e}")
    
    # Save statistics
    stats_file = os.path.join(viz_dir, "trajectory_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Generated trajectory statistics: trajectory_statistics.json")


def _create_trajectory_dashboard_index(viz_dir: str, plot_files: List[str], 
                                     shifts_by_type: Dict[str, Dict[str, str]]) -> str:
    """Create an HTML dashboard index for trajectory analysis."""
    
    dashboard_file = os.path.join(viz_dir, "trajectory_analysis_dashboard.html")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trajectory Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
            .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
            .plot-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #f9f9f9;
                text-align: center;
            }
            .plot-card h3 { margin-top: 0; color: #333; }
            .plot-card a {
                display: inline-block;
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 5px;
            }
            .plot-card a:hover { background-color: #0056b3; }
            .stats-section { margin: 20px 0; }
            .shift-types { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
            .shift-type-badge {
                background-color: #28a745;
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 0.9em;
            }
            .description { color: #666; margin: 10px 0; line-height: 1.5; }
        </style>
    </head>
    <body>
        <h1>🛩️ Trajectory Analysis Dashboard</h1>
        
        <div class="description">
            <p><strong>Interactive trajectory analysis</strong> showing how parameter shifts affect aircraft movement patterns.</p>
            <p>Each plot shows normalized trajectories (relative to baseline starting positions) to highlight movement deviations.</p>
        </div>
        
        <h2>📊 Analysis Plots</h2>
        <div class="dashboard-grid">
    """
    
    # Add cards for each plot
    shift_type_descriptions = {
        'speed': 'Speed variations (±5-30 kt) showing acceleration/deceleration effects',
        'pos_closer': 'Position shifts moving agents closer together',
        'pos_lateral': 'Lateral position deviations creating crossing scenarios',
        'hdg': 'Heading changes (±5-30°) creating converging/diverging paths',
        'heading': 'Heading changes (±5-30°) creating converging/diverging paths',
        'aircraft': 'Aircraft type changes (B737, B747, CRJ9) affecting performance characteristics',
        'waypoint': 'Waypoint destination shifts (±0.05-0.2°) changing flight objectives'
    }
    
    for plot_file in plot_files:
        plot_name = os.path.basename(plot_file)
        
        if 'all_shifts' in plot_name:
            title = "All Parameter Shifts"
            description = "Combined view of all shift types and their effects on trajectory patterns"
            shift_types = list(shifts_by_type.keys())
        elif 'multi_agent_combined' in plot_name:
            title = "Multi-Agent Combined View"
            description = "Single graph showing all agents with different shift types simultaneously for direct comparison"
            shift_types = list(shifts_by_type.keys())
        else:
            # Extract shift type from filename
            shift_type = plot_name.replace('trajectory_analysis_', '').replace('.html', '')
            title = f"{shift_type.replace('_', ' ').title()} Shifts"
            description = shift_type_descriptions.get(shift_type, f"Analysis of {shift_type} parameter variations")
            shift_types = [shift_type]
        
        html_content += f"""
            <div class="plot-card">
                <h3>{title}</h3>
                <p>{description}</p>
                <div class="shift-types">
        """
        
        for stype in shift_types:
            html_content += f'<span class="shift-type-badge">{stype}</span>'
        
        html_content += f"""
                </div>
                <a href="./{plot_name}" target="_blank">View Analysis</a>
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>📈 Key Insights</h2>
        <div class="description">
            <ul>
                <li><strong>Speed Shifts:</strong> Affect trajectory timing and path completion rates</li>
                <li><strong>Position Shifts:</strong> Create spatial conflicts and alter separation distances</li>
                <li><strong>Heading Shifts:</strong> Generate convergent/divergent flight paths</li>
                <li><strong>Aircraft Type Shifts:</strong> Change performance characteristics (B737, B747, CRJ9)</li>
                <li><strong>Waypoint Shifts:</strong> Alter destination objectives and flight planning</li>
                <li><strong>Color Coding:</strong> Each shift type has distinct colors, with target agents shown prominently</li>
                <li><strong>Waypoint Markers:</strong> Gold stars show destination waypoints for each agent</li>
                <li><strong>Normalization:</strong> All trajectories are shown relative to baseline starting positions</li>
            </ul>
        </div>
        
        <h2>🔍 How to Use</h2>
        <div class="description">
            <ol>
                <li><strong>Click on analysis links</strong> to open interactive Plotly visualizations</li>
                <li><strong>Hover over trajectories</strong> to see detailed information about each point</li>
                <li><strong>Use zoom and pan</strong> to examine specific trajectory sections</li>
                <li><strong>Compare colors</strong> to distinguish between shift types and agents</li>
                <li><strong>Look for patterns</strong> in how different shifts affect each agent's movement</li>
            </ol>
        </div>
        
        <hr>
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Generated by trajectory_comparison_plot.py as part of targeted shift analysis
        </p>
    </body>
    </html>
    """
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return dashboard_file