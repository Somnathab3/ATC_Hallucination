#!/usr/bin/env python3
"""
Matplotlib visualization script for ATC conflict scenarios.
Generates detailed plots showing aircraft positions, headings, waypoints, and information.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import os

def load_scenario(scenario_path):
    """Load scenario data from JSON file."""
    with open(scenario_path, 'r') as f:
        return json.load(f)

def draw_aircraft(ax, agent, color, show_info=True):
    """Draw aircraft as large triangle pointing in heading direction."""
    lat, lon = agent['lat'], agent['lon']
    hdg = agent['hdg_deg']
    agent_id = agent['id']
    
    # Aircraft triangle size - much larger
    size = 0.08  # degrees in lat/lon
    
    # Create triangle pointing north (0 degrees)
    triangle_points = np.array([
        [0, size*1.5],      # nose (pointing north)
        [-size*0.8, -size*0.7],   # left wing
        [size*0.8, -size*0.7]     # right wing
    ])
    
    # Rotate triangle to match heading (clockwise from north)
    angle_rad = np.radians(-hdg)  # negative because we want clockwise rotation
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_points = triangle_points @ rotation_matrix.T
    
    # Translate to aircraft position
    rotated_points[:, 0] += lon
    rotated_points[:, 1] += lat
    
    # Draw aircraft
    triangle = patches.Polygon(rotated_points, closed=True, 
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(triangle)
    
    if show_info:
        # Add aircraft information text box
        info_text = (f"{agent_id}\n"
                    f"HDG: {hdg:.0f}°\n"
                    f"SPD: {agent['spd_kt']:.0f}kt\n"
                    f"ALT: {agent['alt_ft']:.0f}ft\n"
                    f"LAT: {lat:.3f}°\n"
                    f"LON: {lon:.3f}°")
        
        # Position text boxes strategically around the plot
        text_positions = {
            'A1': (-0.4, 0.3),   # Top left
            'A2': (0.4, 0.3),    # Top right  
            'A3': (0.4, -0.3),   # Bottom right
            'A4': (-0.4, -0.3),  # Bottom left
            'A0': (-0.4, 0.3),   # For canonical crossing
        }
        
        offset_x, offset_y = text_positions.get(agent_id, (0.3, 0.3))
        
        ax.text(lon + offset_x, lat + offset_y, info_text,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor='white', alpha=0.9, edgecolor='black'),
                ha='left' if offset_x > 0 else 'right',
                va='bottom' if offset_y > 0 else 'top')

def draw_waypoint(ax, waypoint, agent_id, color):
    """Draw waypoint as diamond with information."""
    wp_lat, wp_lon = waypoint['lat'], waypoint['lon']
    
    # Draw waypoint as diamond - smaller than aircraft
    diamond = patches.RegularPolygon((wp_lon, wp_lat), 4, radius=0.03,
                                   orientation=np.pi/4, facecolor=color, 
                                   edgecolor='black', alpha=0.8, linewidth=1.5)
    ax.add_patch(diamond)
    
    # Add waypoint label
    wp_text = f"WP-{agent_id}\nLAT: {wp_lat:.3f}°\nLON: {wp_lon:.3f}°"
    
    # Position waypoint labels strategically
    wp_positions = {
        'A1': (0.15, 0.1),   
        'A2': (-0.25, 0.1),    
        'A3': (-0.25, -0.1),   
        'A4': (0.15, -0.1),  
        'A0': (0.15, 0.1),   
    }
    
    wp_offset_x, wp_offset_y = wp_positions.get(agent_id, (0.15, 0.1))
    
    ax.text(wp_lon + wp_offset_x, wp_lat + wp_offset_y, wp_text,
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor=color, alpha=0.6, edgecolor='black'),
            ha='left' if wp_offset_x > 0 else 'right', 
            va='bottom' if wp_offset_y > 0 else 'top')

def draw_flight_path(ax, agent, color):
    """Draw dashed line from aircraft to its waypoint."""
    if 'waypoint' in agent:
        start_lat, start_lon = agent['lat'], agent['lon']
        wp_lat, wp_lon = agent['waypoint']['lat'], agent['waypoint']['lon']
        
        ax.plot([start_lon, wp_lon], [start_lat, wp_lat], 
                color=color, linestyle='--', alpha=0.7, linewidth=2)

def plot_scenario(scenario_file, output_dir="scenario_plots"):
    """Create visualization for a single scenario."""
    scenario = load_scenario(scenario_file)
    scenario_name = scenario['scenario_name']
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Colors for different aircraft - using distinct colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Process each aircraft
    for i, agent in enumerate(scenario['agents']):
        color = colors[i % len(colors)]
        
        # Draw flight path first (so it appears behind aircraft)
        draw_flight_path(ax, agent, color)
        
        # Draw waypoint if it exists
        if 'waypoint' in agent:
            draw_waypoint(ax, agent['waypoint'], agent['id'], color)
        
        # Draw aircraft
        draw_aircraft(ax, agent, color)
    
    # Draw center point
    center = scenario['center']
    ax.plot(center['lon'], center['lat'], 'ko', markersize=10, markeredgewidth=2)
    ax.text(center['lon'] + 0.05, center['lat'] + 0.05, 
            f"Center\nLAT: {center['lat']:.1f}°\nLON: {center['lon']:.1f}°",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='yellow', alpha=0.9, edgecolor='black'),
            ha='left', va='bottom')
    
    # Set up the plot styling
    ax.set_xlabel('Longitude (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (degrees)', fontsize=14, fontweight='bold')
    
    # Create title
    title_name = scenario_name.replace('_', ' ').title()
    title = f'{title_name} Scenario'
    notes = scenario.get("notes", "")
    ax.set_title(f'{title}\n{notes}', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=12, label='Aircraft', markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
               markersize=10, label='Waypoint', markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, 
               alpha=0.7, label='Flight Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=10, label='Sector Center')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
              fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Calculate appropriate axis limits with some padding
    all_lats = [agent['lat'] for agent in scenario['agents']]
    all_lons = [agent['lon'] for agent in scenario['agents']]
    
    if any('waypoint' in agent for agent in scenario['agents']):
        all_lats.extend([agent['waypoint']['lat'] for agent in scenario['agents'] if 'waypoint' in agent])
        all_lons.extend([agent['waypoint']['lon'] for agent in scenario['agents'] if 'waypoint' in agent])
    
    # Add center to the bounds
    all_lats.append(center['lat'])
    all_lons.append(center['lon'])
    
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    
    # Ensure minimum range for visibility
    if lat_range < 0.5:
        lat_range = 0.5
    if lon_range < 0.5:
        lon_range = 0.5
        
    lat_margin = lat_range * 0.3
    lon_margin = lon_range * 0.3
    
    ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    # Add scenario metadata box
    metadata_text = (f"Simulation dt: {scenario.get('sim_dt_s', 'N/A')}s\n"
                    f"Seed: {scenario.get('seed', 'N/A')}\n"
                    f"Aircraft count: {len(scenario['agents'])}")
    
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", 
                                facecolor='lightblue', alpha=0.9, edgecolor='black'),
            verticalalignment='top', fontweight='bold')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{scenario_name}_plot.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Generated plot: {output_file}")
    return output_file

def visualize_all_scenarios(scenarios_dir="scenarios", output_dir="scenario_plots"):
    """Generate visualizations for all scenario JSON files."""
    scenarios_path = Path(scenarios_dir)
    
    if not scenarios_path.exists():
        print(f"Scenarios directory '{scenarios_dir}' not found!")
        return
    
    json_files = list(scenarios_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{scenarios_dir}'")
        return
    
    print(f"Found {len(json_files)} scenario files:")
    for json_file in json_files:
        print(f"  - {json_file.name}")
    
    print(f"\nGenerating visualizations...")
    generated_plots = []
    
    for json_file in sorted(json_files):
        try:
            plot_file = plot_scenario(json_file, output_dir)
            generated_plots.append(plot_file)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nCompleted! Generated {len(generated_plots)} plots in '{output_dir}/':")
    for plot in generated_plots:
        print(f"  - {os.path.basename(plot)}")

if __name__ == "__main__":
    # Generate all scenario visualizations
    visualize_all_scenarios()