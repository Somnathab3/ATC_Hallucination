"""
Plot Training Rewards Grouped by Complexity
============================================
Creates separate plots for:
- 2×2 scenarios (3 lines: chase, cross, merge)
- 3+1 scenarios (3 lines: chase, cross, merge)
- 4-all scenarios (3 lines: chase, cross, merge)
- Generic scenario (1 line)

Shows maximum reward values with annotations.
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import re

# Configuration
TRAINING_DIR = Path(r"F:\ATC_Hallucination\training")
OUTPUT_DIR = Path(r"F:\ATC_Hallucination\results\figs_plotly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette by scenario family
COLORS = {
    'chase': '#1f77b4',  # Blue
    'cross': '#ff7f0e',  # Orange
    'merge': '#2ca02c',  # Green
    'generic': '#9467bd' # Purple
}

def extract_scenario_info(folder_name):
    """
    Extract scenario family and complexity from folder name.
    Returns: (family, complexity, display_name)
    Example: 'results_PPO_chase_2x2_20251008_015945' -> ('chase', '2x2', 'Chase')
    """
    # Remove timestamp and prefix
    name = folder_name.replace('results_PPO_', '')
    name = re.sub(r'_\d{8}_\d{6}$', '', name)  # Remove _YYYYMMDD_HHMMSS
    
    parts = name.split('_')
    
    if len(parts) == 1 and parts[0] == 'generic':
        # Generic case
        return ('generic', 'generic', 'Generic')
    elif len(parts) == 2:
        # chase_2x2, cross_3p1, merge_4all
        family = parts[0]
        complexity = parts[1]
        display_name = family.capitalize()
        return (family, complexity, display_name)
    
    return (None, None, None)

def discover_training_data():
    """
    Discover all training folders and group by complexity.
    Returns: dict with keys '2x2', '3p1', '4all', 'generic'
    """
    grouped = {
        '2x2': [],
        '3p1': [],
        '4all': [],
        'generic': []
    }
    
    for folder in TRAINING_DIR.glob("results_PPO_*"):
        if not folder.is_dir():
            continue
        
        family, complexity, display_name = extract_scenario_info(folder.name)
        if family is None:
            continue
        
        csv_path = folder / "training_progress.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        # Calculate training time from timestamps (ts column is Unix timestamp in seconds)
        if 'ts' in df.columns and len(df) > 0:
            training_time_s = df['ts'].iloc[-1] - df['ts'].iloc[0]
        else:
            training_time_s = 0
        
        # Find maximum reward
        max_idx = df['reward_mean'].idxmax()
        max_reward = df.loc[max_idx, 'reward_mean']
        max_timestep = df.loc[max_idx, 'steps_sampled']
        
        # Find step to perfect band (zero_conflict_streak >= 5)
        perfect_step = None
        if 'zero_conflict_streak' in df.columns:
            perfect_rows = df[df['zero_conflict_streak'] >= 5]
            if len(perfect_rows) > 0:
                perfect_step = perfect_rows.iloc[0]['steps_sampled']
        
        data_entry = {
            'family': family,
            'display_name': display_name,
            'folder': folder,
            'df': df,
            'color': COLORS.get(family, '#808080'),
            'max_reward': max_reward,
            'max_timestep': max_timestep,
            'training_time_s': training_time_s,
            'perfect_step': perfect_step,
            'final_reward': df['reward_mean'].iloc[-1],
            'total_steps': df['steps_sampled'].iloc[-1],
            'total_episodes': df['episodes'].iloc[-1]
        }
        
        if complexity in grouped:
            grouped[complexity].append(data_entry)
    
    return grouped

def create_plot_for_group(group_data, group_name, title_suffix):
    """Create a plotly figure for a specific complexity group."""
    fig = go.Figure()
    
    print(f"\n{group_name} Scenarios:")
    print("=" * 70)
    
    # Text positions to avoid overlap (cycle through different positions)
    # Use a set of text positions that alternate and avoid overlap with lines.
    # We'll cycle through: 'top right', 'bottom right', 'top left', 'bottom left', 'middle right', 'middle left'
    # and skip positions that would overlap with the line at the max point.
    text_positions_all = [
        'top right', 'bottom right', 'top left', 'bottom left', 'middle right', 'middle left'
    ]
    
    # Define shifts for positioning annotations to match text positions
    shifts = {
        'top right': {'xshift': 10, 'yshift': 10, 'xanchor': 'left', 'yanchor': 'bottom'},
        'bottom right': {'xshift': 10, 'yshift': -10, 'xanchor': 'left', 'yanchor': 'top'},
        'top left': {'xshift': -10, 'yshift': 10, 'xanchor': 'right', 'yanchor': 'bottom'},
        'bottom left': {'xshift': -10, 'yshift': -10, 'xanchor': 'right', 'yanchor': 'top'},
        'middle right': {'xshift': 10, 'yshift': 0, 'xanchor': 'left', 'yanchor': 'middle'},
        'middle left': {'xshift': -10, 'yshift': 0, 'xanchor': 'right', 'yanchor': 'middle'},
    }
    text_positions = []
    used_positions = set()
    for i in range(len(group_data)):
        # Cycle through positions, but avoid reusing the same for adjacent lines
        pos = text_positions_all[i % len(text_positions_all)]
        # If already used, pick next available
        j = 0
        while pos in used_positions and j < len(text_positions_all):
            i2 = (i + j) % len(text_positions_all)
            pos = text_positions_all[i2]
            j += 1
        text_positions.append(pos)
        used_positions.add(pos)

    for idx, entry in enumerate(group_data):
        df = entry['df']
        timesteps = df['steps_sampled'].values
        rewards = df['reward_mean'].values
        
        # Main trace
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=rewards,
            mode='lines',
            name=entry['display_name'],
            line=dict(color=entry['color'], width=2.5),
            hovertemplate=f'<b>{entry["display_name"]}</b><br>' +
                          'Timestep: %{x:,.0f}<br>' +
                          'Reward: %{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Add marker for maximum reward (marker only) and an annotation box with dark background
        text_pos = text_positions[idx % len(text_positions)]
        shift = shifts.get(text_pos, {'xshift': 0, 'yshift': 0, 'xanchor': 'center', 'yanchor': 'middle'})

        # Marker only (no inline text) so we can use an annotation with filled background
        fig.add_trace(go.Scatter(
            x=[entry['max_timestep']],
            y=[entry['max_reward']],
            mode='markers',
            name=f'{entry["display_name"]} (Max)',
            marker=dict(
                color=entry['color'],
                size=14,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hovertemplate=f'<b>{entry["display_name"]} - Maximum</b><br>' +
                          f'Timestep: {entry["max_timestep"]:,.0f}<br>' +
                          f'Reward: {entry["max_reward"]:.2f}<br>' +
                          '<extra></extra>'
        ))

        # Add annotation with dark background and white font for clear visibility
        ann_text = f"Max {entry['max_reward']:.1f}"
        fig.add_annotation(
            x=entry['max_timestep'],
            y=entry['max_reward'],
            text=f"<b>{ann_text}</b>",
            showarrow=False,
            font=dict(size=11, color='white', family='Arial Black'),
            bgcolor=entry['color'],
            bordercolor='rgba(0,0,0,0.25)',
            borderwidth=1,
            borderpad=4,
            xshift=shift.get('xshift', 0),
            yshift=shift.get('yshift', 0),
            xanchor=shift.get('xanchor', 'center'),
            yanchor=shift.get('yanchor', 'middle')
        )
        
        print(f"{entry['display_name']:10s} | Max Reward: {entry['max_reward']:>7.2f} @ {entry['max_timestep']:>10,} steps | "
              f"Training Time: {entry['training_time_s']:>6.1f}s | Perfect Band: {entry['perfect_step'] if entry['perfect_step'] else 'N/A'}")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Training Rewards: {title_suffix}',
            font=dict(size=18, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Training Timesteps', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='#ecf0f1',
            dtick=200000,
            tickformat='.2s',
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text='Mean Episode Reward', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#95a5a6',
            zerolinewidth=1.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=dict(text='Scenario', font=dict(size=12)),
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top'
        ),
        width=1000,
        height=600,
        margin=dict(l=80, r=40, t=80, b=80)
    )
    
    return fig

def save_as_png(fig, filename):
    """Save plotly figure as PNG using kaleido."""
    try:
        fig.write_image(str(OUTPUT_DIR / filename), width=1000, height=600, scale=2)
        print(f"  ✓ Saved PNG: {filename}")
    except Exception as e:
        print(f"  ✗ Could not save PNG (kaleido not installed?): {e}")

def main():
    """Main execution."""
    print("=" * 70)
    print("TRAINING REWARDS VISUALIZATION - GROUPED BY COMPLEXITY")
    print("=" * 70)
    
    # Discover all training data
    grouped = discover_training_data()
    
    # Create plots for each group
    group_configs = [
        ('2x2', '2×2 Scenarios (2 vs 2 Aircraft)'),
        ('3p1', '3+1 Scenarios (3 vs 1 Aircraft)'),
        ('4all', '4-all Scenarios (4 Aircraft, All Active)'),
        ('generic', 'Generic Scenario (Mixed Traffic)')
    ]
    
    for group_key, title_suffix in group_configs:
        group_data = grouped[group_key]
        
        if not group_data:
            print(f"\nNo data found for {group_key}")
            continue
        
        # Create figure
        fig = create_plot_for_group(group_data, group_key, title_suffix)
        
        # Save HTML
        html_filename = f"fig5_1_training_{group_key}.html"
        fig.write_html(str(OUTPUT_DIR / html_filename))
        print(f"  ✓ Saved HTML: {html_filename}")
        
        # Save PNG
        png_filename = f"fig5_1_training_{group_key}.png"
        save_as_png(fig, png_filename)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Create summary table
    summary_rows = []
    for group_key in ['2x2', '3p1', '4all', 'generic']:
        for entry in grouped[group_key]:
            summary_rows.append({
                'Scenario': f"{entry['display_name']} {group_key}",
                'Max Reward': entry['max_reward'],
                'Max @ Step': entry['max_timestep'],
                'Final Reward': entry['final_reward'],
                'Total Steps': entry['total_steps'],
                'Training Time (s)': entry['training_time_s'],
                'Perfect Band Step': entry['perfect_step'] if entry['perfect_step'] else 'N/A'
            })
    
    # Save summary CSV
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR.parent.parent / "results" / "tables" / "training_summary_grouped.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary table saved: {summary_csv}")
    
    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
