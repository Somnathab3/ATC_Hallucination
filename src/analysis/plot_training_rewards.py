"""
Plot Training Rewards Across All Scenarios
Visualizes training progress with rewards over timesteps for all trained models.
Creates separate plots by training complexity: 2×2, 3+1, 4-all, and Generic.
Automatically discovers training folders and extracts scenario names.
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Base directory for training results
TRAINING_DIR = Path(r"F:\ATC_Hallucination\training")
OUTPUT_DIR = Path(r"F:\ATC_Hallucination\results\figs_plotly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for scenarios (by family)
COLOR_PALETTE = {
    'chase': "#1f77b4",  # Blue
    'cross': "#ff7f0e",  # Orange
    'merge': "#2ca02c",  # Green
    'generic': "#9467bd" # Purple
}

def extract_scenario_name(folder_name):
    """
    Extract scenario name from folder name like 'results_PPO_chase_2x2_20251008_015945'.
    Converts underscores to spaces and capitalizes words.
    """
    # Pattern: results_PPO_{scenario_name}_{timestamp}
    match = re.match(r'results_PPO_(.+?)_\d{8}_\d{6}', folder_name)
    if match:
        scenario = match.group(1)
        # Convert underscores to spaces and capitalize each word
        scenario_display = scenario.replace('_', ' ').title()
        return scenario_display
    return folder_name

def discover_training_folders():
    """
    Automatically discover all training result folders in the training directory.
    Returns dict of {scenario_name: folder_path}.
    """
    scenarios = {}
    
    if not TRAINING_DIR.exists():
        print(f"Warning: Training directory not found: {TRAINING_DIR}")
        return scenarios
    
    # Find all folders matching the pattern results_PPO_*
    for folder in TRAINING_DIR.glob("results_PPO_*"):
        if folder.is_dir():
            scenario_name = extract_scenario_name(folder.name)
            scenarios[scenario_name] = str(folder)
    
    return scenarios

def assign_colors(scenarios):
    """Assign colors to scenarios from the color palette."""
    scenario_names = sorted(scenarios.keys())  # Sort for consistency
    colors = {}
    for idx, scenario in enumerate(scenario_names):
        colors[scenario] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    return colors

def load_training_data(folder_path):
    """Load training progress CSV file."""
    csv_path = Path(folder_path) / "training_progress.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found!")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def create_combined_plot(scenarios, colors):
    """Create combined Plotly figure with all scenarios."""
    fig = go.Figure()
    
    max_reward_info = {}  # Store max reward info for annotations
    
    # Load and plot data for each scenario
    for scenario_name, folder_path in scenarios.items():
        print(f"Loading data for {scenario_name}...")
        df = load_training_data(folder_path)
        
        if df is None:
            continue
        
        timesteps = df['steps_sampled'].values
        rewards = df['reward_mean'].values
        
        # Find maximum reward and its timestep
        max_idx = np.argmax(rewards)
        max_reward = rewards[max_idx]
        max_timestep = timesteps[max_idx]
        
        max_reward_info[scenario_name] = {
            'timestep': max_timestep,
            'reward': max_reward,
            'color': colors[scenario_name]
        }
        
        # Add main trace for the scenario
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=rewards,
            mode='lines',
            name=scenario_name,
            line=dict(color=colors[scenario_name], width=2),
            hovertemplate=f'<b>{scenario_name}</b><br>' +
                          'Timestep: %{x:,.0f}<br>' +
                          'Reward: %{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Add marker for maximum reward
        fig.add_trace(go.Scatter(
            x=[max_timestep],
            y=[max_reward],
            mode='markers+text',
            name=f'{scenario_name} (Max)',
            marker=dict(
                color=colors[scenario_name],
                size=12,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            text=[f'{max_reward:.2f}'],
            textposition='top center',
            textfont=dict(size=10, color=colors[scenario_name]),
            showlegend=False,
            hovertemplate=f'<b>{scenario_name} - Maximum</b><br>' +
                          f'Timestep: {max_timestep:,.0f}<br>' +
                          f'Reward: {max_reward:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Add vertical line at max reward
        fig.add_vline(
            x=max_timestep,
            line=dict(color=colors[scenario_name], width=1, dash='dash'),
            opacity=0.3,
            annotation_text=f"{max_timestep:,.0f}",
            annotation_position="bottom",
            annotation_font_size=8,
            annotation_font_color=colors[scenario_name]
        )
    
    # Update layout with custom x-axis formatting
    fig.update_layout(
        title=dict(
            text='Training Rewards Across All Scenarios',
            font=dict(size=20, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Training Timesteps', font=dict(size=14, color='#34495e')),
            tickfont=dict(size=12),
            gridcolor='#ecf0f1',
            # Custom tick formatting for 2×10^5 intervals
            dtick=200000,  # 2×10^5
            tickformat='.2s',  # Scientific notation
            tickmode='linear',
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text='Mean Episode Reward', font=dict(size=14, color='#34495e')),
            tickfont=dict(size=12),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#95a5a6',
            zerolinewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=dict(text='Scenario', font=dict(size=12)),
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        width=1400,
        height=700,
        margin=dict(l=80, r=200, t=80, b=80)
    )
    
    # Print summary of max rewards
    print("\n" + "="*60)
    print("MAXIMUM REWARDS SUMMARY")
    print("="*60)
    for scenario, info in max_reward_info.items():
        print(f"{scenario:25s} | Timestep: {info['timestep']:>10,} | Max Reward: {info['reward']:>8.2f}")
    print("="*60)
    
    return fig

def main():
    """Main execution function."""
    print("Creating combined training rewards plot...")
    print("Discovering training folders...\n")
    
    # Discover all training folders
    scenarios = discover_training_folders()
    
    if not scenarios:
        print("Error: No training folders found!")
        print(f"Please check that training results exist in: {TRAINING_DIR}")
        return
    
    print(f"Found {len(scenarios)} training scenarios:")
    for scenario_name, folder_path in sorted(scenarios.items()):
        print(f"  - {scenario_name}: {Path(folder_path).name}")
    print()
    
    # Assign colors to scenarios
    colors = assign_colors(scenarios)
    
    # Create the plot
    fig = create_combined_plot(scenarios, colors)
    
    # Save as interactive HTML
    output_file = "training_rewards_combined.html"
    fig.write_html(output_file)
    print(f"\n✓ Interactive plot saved to: {output_file}")
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    main()
