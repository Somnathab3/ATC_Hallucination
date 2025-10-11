"""
Visual summary of scenario geometries with TCPA/DCPA annotations.
"""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any

def latlon_to_xy(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple:
    """Convert lat/lon to x/y in NM relative to center."""
    x_nm = (lon - center_lon) * 60.0 * math.cos(math.radians(center_lat))
    y_nm = (lat - center_lat) * 60.0
    return x_nm, y_nm

def calculate_tcpa_dcpa(ag1: Dict, ag2: Dict) -> tuple:
    """Calculate TCPA and DCPA between two agents."""
    lat1, lon1, hdg1, spd1 = ag1["lat"], ag1["lon"], ag1["hdg_deg"], ag1["spd_kt"]
    lat2, lon2, hdg2, spd2 = ag2["lat"], ag2["lon"], ag2["hdg_deg"], ag2["spd_kt"]
    
    mean_lat = (lat1 + lat2) / 2.0
    rx_nm = (lon2 - lon1) * 60.0 * math.cos(math.radians(mean_lat))
    ry_nm = (lat2 - lat1) * 60.0
    
    spd1_nm_s = spd1 / 3600.0
    spd2_nm_s = spd2 / 3600.0
    
    vx1 = spd1_nm_s * math.sin(math.radians(hdg1))
    vy1 = spd1_nm_s * math.cos(math.radians(hdg1))
    vx2 = spd2_nm_s * math.sin(math.radians(hdg2))
    vy2 = spd2_nm_s * math.cos(math.radians(hdg2))
    
    vx_nm_s = vx2 - vx1
    vy_nm_s = vy2 - vy1
    
    v_mag_sq = vx_nm_s**2 + vy_nm_s**2
    
    if v_mag_sq < 1e-10:
        return None, None
    
    r_dot_v = rx_nm * vx_nm_s + ry_nm * vy_nm_s
    tcpa_raw = -r_dot_v / v_mag_sq
    tcpa_s = max(0.0, min(500.0, tcpa_raw))
    
    rx_at_cpa = rx_nm + vx_nm_s * tcpa_s
    ry_at_cpa = ry_nm + vy_nm_s * tcpa_s
    dcpa_nm = math.sqrt(rx_at_cpa**2 + ry_at_cpa**2)
    
    return tcpa_s, dcpa_nm

def plot_scenario(ax, scenario_path: Path, title: str):
    """Plot a single scenario with TCPA/DCPA annotations."""
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    
    center_lat = scenario['center']['lat']
    center_lon = scenario['center']['lon']
    agents = scenario['agents']
    
    # Colors for agents
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # Plot agents and their trajectories
    for i, agent in enumerate(agents):
        x, y = latlon_to_xy(agent['lat'], agent['lon'], center_lat, center_lon)
        wx, wy = latlon_to_xy(agent['waypoint']['lat'], agent['waypoint']['lon'], 
                              center_lat, center_lon)
        
        # Agent position
        ax.scatter(x, y, s=200, c=colors[i], marker='o', edgecolors='black', 
                  linewidths=2, zorder=10, label=agent['id'])
        
        # Trajectory line
        ax.arrow(x, y, wx-x, wy-y, head_width=1.5, head_length=2, 
                fc=colors[i], ec=colors[i], alpha=0.3, linewidth=2, zorder=1)
        
        # Heading indicator
        hdg_rad = math.radians(agent['hdg_deg'])
        dx = 5 * math.sin(hdg_rad)
        dy = 5 * math.cos(hdg_rad)
        ax.arrow(x, y, dx, dy, head_width=1, head_length=1.5, 
                fc=colors[i], ec='black', linewidth=1.5, zorder=11)
        
        # Waypoint
        ax.scatter(wx, wy, s=100, c=colors[i], marker='s', alpha=0.5, 
                  edgecolors='black', linewidths=1, zorder=5)
    
    # Calculate and annotate conflicts
    conflict_count = 0
    for i, ag1 in enumerate(agents):
        for j, ag2 in enumerate(agents[i+1:], start=i+1):
            tcpa, dcpa = calculate_tcpa_dcpa(ag1, ag2)
            if tcpa and tcpa > 0 and dcpa < 5.0:
                conflict_count += 1
                # Draw conflict indicator
                x1, y1 = latlon_to_xy(ag1['lat'], ag1['lon'], center_lat, center_lon)
                x2, y2 = latlon_to_xy(ag2['lat'], ag2['lon'], center_lat, center_lon)
                ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.3, linewidth=1)
    
    # Center point
    ax.scatter(0, 0, s=50, c='gray', marker='+', linewidths=2, zorder=3)
    
    ax.set_xlabel('East-West (NM)', fontsize=10)
    ax.set_ylabel('North-South (NM)', fontsize=10)
    ax.set_title(f'{title}\n({conflict_count} conflicts)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def main():
    scenarios_dir = Path('scenarios')
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('ATC Hallucination Detection: Scenario Geometries\n' +
                'Difficulty progression: 2x2 (Easy) → 3p1 (Medium) → 4all (Hard)',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Scenario layout
    scenarios = [
        ('merge_2x2.json', 'Merge 2x2 (Easy)', 1),
        ('merge_3p1.json', 'Merge 3p1 (Medium)', 2),
        ('merge_4all.json', 'Merge 4all (Hard)', 3),
        ('cross_2x2.json', 'Cross 2x2 (Easy)', 4),
        ('cross_3p1.json', 'Cross 3p1 (Medium)', 5),
        ('cross_4all.json', 'Cross 4all (Hard)', 6),
        ('chase_2x2.json', 'Chase 2x2 (Easy)', 7),
        ('chase_3p1.json', 'Chase 3p1 (Medium)', 8),
        ('chase_4all.json', 'Chase 4all (Hard)', 9),
    ]
    
    for scenario_file, title, subplot_idx in scenarios:
        ax = fig.add_subplot(3, 3, subplot_idx)
        plot_scenario(ax, scenarios_dir / scenario_file, title)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = 'plots/scenario_geometries_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also save as PDF for thesis
    output_path_pdf = 'plots/scenario_geometries_summary.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")
    
    plt.show()

if __name__ == '__main__':
    main()
