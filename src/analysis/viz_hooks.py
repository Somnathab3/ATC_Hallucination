"""
Module Name: viz_hooks.py
Description: 
    Visualization integration hooks for shift testing pipeline.
    Provides seamless integration between robustness testing and visualization components.
    
    Architecture:
        Acts as glue layer between:
            - Testing modules (intrashift_tester, intershift_matrix)
            - Environment outputs (trajectory CSV/JSON)
            - Visualization modules (trajectory_comparison_plot, trajectory_comparison_map)
            - Analysis modules (hallucination_detector_enhanced)
    
    Automatic Artifact Generation:
        Episode-level:
            - Trajectory comparison plots (baseline vs shifted)
            - Geographic maps with conflict zones (Folium/Plotly)
            - Separation time series with LoS markers
            - Hallucination confusion matrices
        
        Run-level:
            - Summary dashboards aggregating all episodes
            - Shift analysis heatmaps (performance degradation)
            - Statistical comparison tables
            - Bundle analysis (multi-shift aggregation)
    
    Data Flow:
        1. Testing module runs episodes → generates trajectory CSV/JSON
        2. viz_hooks extracts trajectory + hallucination series
        3. Merges data into unified DataFrame
        4. Calls visualization modules with standardized format
        5. Saves artifacts to results directory
    
    Key Functions:
        - make_episode_visuals: Per-episode plots and maps
        - make_run_visuals: Aggregate analysis dashboards
        - _series_to_frame: Data format conversion and merging

Author: Som
Date: 2025-10-04
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _series_to_frame(traj_csv: str, series: dict) -> pd.DataFrame:
    """
    Convert trajectory CSV and detector series into merged DataFrame for visualization.
    
    Trajectory CSV Format (from MARLCollisionEnv):
        Core columns:
            - episode_id, step_idx, sim_time_s, agent_id
        
        State columns:
            - lat_deg, lon_deg, alt_ft, hdg_deg, tas_kt, cas_kt
        
        Action columns:
            - action_hdg_delta_deg, action_spd_delta_kt
        
        Reward columns:
            - reward_progress, reward_drift, reward_violate_entry, reward_violate_step,
              reward_act_cost, reward_time, reward_reach, reward_terminal, reward_team,
              reward_total
        
        Separation columns:
            - min_separation_nm, dist_to_{agent}_nm (for each agent)
        
        Conflict columns:
            - conflict_flag, collision_flag, conflict_pairs_count
        
        Waypoint columns:
            - wp_dist_nm, waypoint_reached, waypoint_hits
        
        Hallucination columns (if enabled):
            - gt_conflict, predicted_alert, tp, fp, fn, tn
    
    Series Data (from HallucinationDetector):
        Optional additional hallucination metrics if not already in CSV.
        Contains time-series arrays: gt_conflict, alert, tp, fp, fn, tn.
    
    Merging Strategy:
        1. Load trajectory CSV with full environment state
        2. Validate required visualization columns present
        3. Add missing columns with sensible defaults if needed
        4. Merge series data if not already in CSV
    
    Args:
        traj_csv: Path to trajectory CSV file (rich format from environment).
        series: Series data dict from hallucination detector (optional).
    
    Returns:
        Merged pd.DataFrame with standardized column names for visualization.
        Empty DataFrame if CSV missing or invalid.
    """
    if not os.path.exists(traj_csv):
        print(f"Warning: Trajectory CSV not found: {traj_csv}")
        return pd.DataFrame()
        
    df = pd.read_csv(traj_csv)
    
    if df.empty:
        print(f"Warning: Empty CSV file: {traj_csv}")
        return df
    
    # The rich CSV already has most columns we need, just add column mapping
    column_mapping = {
        "lat_deg": "lat_deg",  # Already correct
        "lon_deg": "lon_deg",  # Already correct
        "hdg_deg": "hdg_deg",  # Already correct  
        "tas_kt": "tas_kt",    # Already correct
        "sim_time_s": "sim_time_s",  # Already correct
        "agent_id": "agent_id",      # Already correct
        "min_separation_nm": "min_separation_nm",  # Already correct
        "conflict_flag": "conflict_flag",          # Already correct
        "step_idx": "step_idx"                     # Already correct
    }
    
    # The rich CSV already has the right column names and structure!
    # Just ensure we have the required visualization columns
    required_viz_columns = ["lat_deg", "lon_deg", "hdg_deg", "tas_kt", "sim_time_s", 
                           "agent_id", "step_idx", "min_separation_nm", "conflict_flag"]
    
    missing_cols = [col for col in required_viz_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required visualization columns: {missing_cols}")
        # Add missing columns with default values
        for col in missing_cols:
            if col in ["lat_deg", "lon_deg"]:
                df[col] = 52.0  # Default coordinates
            elif col in ["hdg_deg"]:
                df[col] = 0.0
            elif col in ["tas_kt"]:
                df[col] = 250.0
            elif col == "sim_time_s":
                df[col] = df.get("t", df.index * 10.0)
            elif col == "agent_id":
                df[col] = "A1"  # Default agent
            elif col == "step_idx":
                df[col] = df.index
            else:
                df[col] = 0
    
    # If series data is provided, it should already be merged in the CSV
    # The environment CSV has conflict_flag, and merged CSV has tp/fp/fn/tn
    if series and "predicted_alert" not in df.columns:
        print("Warning: Series data provided but not found in CSV - series should be pre-merged")
    
    return df


def make_episode_visuals(episode_dir: str, traj_csv: str, detector_series: dict, title: str = "") -> dict:
    """
    Generate comprehensive episode-level visualization artifacts.
    
    Args:
        episode_dir: Directory containing episode data
        traj_csv: Path to trajectory CSV with integrated hallucination data
        detector_series: Series data from hallucination detector
        title: Descriptive title for visualizations
        
    Returns:
        dict: Mapping of visualization types to file paths
    """
    try:
        # Import visualization modules
        from .viz_geographic import build_map
        from .viz_plotly import time_series_panel, animated_geo
        
        # Create viz subdirectory
        viz_dir = os.path.join(episode_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert series to DataFrame
        df = _series_to_frame(traj_csv, detector_series)
        
        if df.empty:
            print(f"Warning: Empty DataFrame for episode visualization in {episode_dir}")
            return {}
        
        # Filter valid data (remove NaN step indices)
        df_clean = df[df["step_idx"].notna()].copy()
        
        # Generate geographic map
        map_html = os.path.join(viz_dir, "map.html")
        try:
            build_map(df_clean, None, out_html=map_html)
        except Exception as e:
            print(f"Warning: Failed to create map for {episode_dir}: {e}")
            map_html = None
        
        # Generate time series plot
        ts_html = os.path.join(viz_dir, "time_series.html")
        try:
            ts_fig = time_series_panel(df_clean, title=f"minSep/DCPA vs time + alerts | {title}")
            ts_fig.write_html(ts_html)
        except Exception as e:
            print(f"Warning: Failed to create time series for {episode_dir}: {e}")
            ts_html = None
        
        # Generate animated plot
        anim_html = os.path.join(viz_dir, "animated_tracks.html")
        try:
            anim_fig = animated_geo(df_clean, title=f"Animated tracks | {title}")
            anim_fig.write_html(anim_html)
        except Exception as e:
            print(f"Warning: Failed to create animation for {episode_dir}: {e}")
            anim_html = None
        
        # Return paths to generated files
        result = {}
        if map_html and os.path.exists(map_html):
            result["map"] = map_html
        if ts_html and os.path.exists(ts_html):
            result["time_series"] = ts_html
        if anim_html and os.path.exists(anim_html):
            result["animated"] = anim_html
            
        return result
        
    except Exception as e:
        print(f"Error generating episode visuals for {episode_dir}: {e}")
        return {}


def make_run_visuals(results_dir: str, scenario_name: str):
    """
    Generate run-level visualization artifacts.
    
    Args:
        results_dir: Directory containing run results
        scenario_name: Name of the scenario
    """
    try:
        # Import visualization modules
        from .viz_matplotlib import plot_degradation_curves, heatmap_agent_vulnerability
        
        # Look for the main summary CSV
        main_csv = os.path.join(results_dir, "targeted_shift_test_summary.csv")
        if not os.path.exists(main_csv):
            print(f"Warning: Main summary CSV not found: {main_csv}")
            return
        
        df = pd.read_csv(main_csv)
        if df.empty:
            print(f"Warning: Empty summary DataFrame in {results_dir}")
            return
        
        # Create viz directory
        fig_dir = os.path.join(results_dir, "viz")
        os.makedirs(fig_dir, exist_ok=True)
        
        # Map common column names to expected names
        column_mapping = {
            'missed_conflict': 'fn_rate',
            'resolution_fail_rate': 'resolution_failure_rate',
            'target_agent': 'agent_id'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Add shift_magnitude column if missing
        if 'shift_magnitude' not in df.columns:
            if 'shift_value' in df.columns:
                df['shift_magnitude'] = df['shift_value'].abs()
            else:
                df['shift_magnitude'] = 1.0  # Default value
        
        # Generate enhanced analysis plots
        _create_enhanced_analysis_plots(df, fig_dir, scenario_name)
        
        # NOTE: Trajectory comparison maps are now generated by the targeted_shift_tester
        # using the newer generate_shift_comparison_maps function, which creates better
        # baseline vs shifted comparison maps in the viz directory
        
        # Generate degradation curves for key metrics
        metrics_to_plot = [
            ('missed_conflict', 'Missed Conflict Rate'),
            ('resolution_fail_rate', 'Resolution Failure Rate'), 
            ('min_CPA_nm', 'Minimum CPA (NM)'),
            ('ghost_conflict', 'Ghost Conflict Rate'),
            ('oscillation_rate', 'Oscillation Rate'),
            ('num_los_events', 'Number of LOS Events'),
            ('min_separation_nm', 'Minimum Separation (NM)')
        ]
        
        for metric_col, metric_title in metrics_to_plot:
            if metric_col in df.columns:
                try:
                    fig = plot_degradation_curves(
                        df, 
                        metric_col=metric_col, 
                        shift_col='shift_magnitude',
                        group_col='shift_type',
                        title=f"{scenario_name}: {metric_title} vs Magnitude"
                    )
                    fig_path = os.path.join(fig_dir, f"deg_{metric_col}.png")
                    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved degradation curve: {fig_path}")
                except Exception as e:
                    print(f"Warning: Failed to create degradation curve for {metric_col}: {e}")
        
        # Generate agent vulnerability heatmap
        if 'agent_id' in df.columns and 'resolution_fail_rate' in df.columns:
            try:
                agents = sorted(df['agent_id'].unique())
                fig = heatmap_agent_vulnerability(
                    df,
                    metric_col='resolution_fail_rate',
                    agents=agents
                )
                fig_path = os.path.join(fig_dir, "agent_vulnerability.png")
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved agent vulnerability heatmap: {fig_path}")
            except Exception as e:
                print(f"Warning: Failed to create agent vulnerability heatmap: {e}")
        
        # Generate safety metrics heatmap if we have LOS data
        if all(col in df.columns for col in ['agent_id', 'num_los_events']):
            try:
                fig = heatmap_agent_vulnerability(
                    df,
                    metric_col='num_los_events', 
                    agents=sorted(df['agent_id'].unique())
                )
                fig_path = os.path.join(fig_dir, "los_events_heatmap.png")
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved LOS events heatmap: {fig_path}")
            except Exception as e:
                print(f"Warning: Failed to create LOS events heatmap: {e}")
        
        print(f"Completed run-level visualizations in {fig_dir}")
        
    except Exception as e:
        print(f"Error generating run visuals for {results_dir}: {e}")


def _create_enhanced_analysis_plots(df: pd.DataFrame, fig_dir: str, scenario_name: str):
    """Create enhanced analysis plots comparing shifts vs various metrics."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create 2x2 subplot for key comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Enhanced Shift Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Shift Magnitude vs LOS Events
        if 'num_los_events' in df.columns and 'shift_magnitude' in df.columns:
            for shift_type in df['shift_type'].unique():
                subset = df[df['shift_type'] == shift_type]
                ax1.scatter(subset['shift_magnitude'], subset['num_los_events'], 
                           label=shift_type, alpha=0.7, s=50)
            ax1.set_xlabel('Shift Magnitude')
            ax1.set_ylabel('Number of LOS Events')
            ax1.set_title('Shift vs Loss of Separation Events')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Shift Magnitude vs Minimum Separation
        if 'min_separation_nm' in df.columns:
            for shift_type in df['shift_type'].unique():
                subset = df[df['shift_type'] == shift_type]
                ax2.scatter(subset['shift_magnitude'], subset['min_separation_nm'], 
                           label=shift_type, alpha=0.7, s=50)
            ax2.set_xlabel('Shift Magnitude')
            ax2.set_ylabel('Minimum Separation (NM)')
            ax2.set_title('Shift vs Minimum Separation Distance')
            ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Safety Threshold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Shift Magnitude vs False Positives  
        if 'fp' in df.columns:
            for shift_type in df['shift_type'].unique():
                subset = df[df['shift_type'] == shift_type]
                ax3.scatter(subset['shift_magnitude'], subset['fp'], 
                           label=shift_type, alpha=0.7, s=50)
            ax3.set_xlabel('Shift Magnitude')
            ax3.set_ylabel('False Positives')
            ax3.set_title('Shift vs False Positive Detections')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Shift Magnitude vs False Negatives
        if 'fn' in df.columns:
            for shift_type in df['shift_type'].unique():
                subset = df[df['shift_type'] == shift_type]
                ax4.scatter(subset['shift_magnitude'], subset['fn'], 
                           label=shift_type, alpha=0.7, s=50)
            ax4.set_xlabel('Shift Magnitude')
            ax4.set_ylabel('False Negatives')
            ax4.set_title('Shift vs False Negative Detections')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the enhanced analysis plot
        enhanced_plot_path = os.path.join(fig_dir, "enhanced_shift_analysis.png")
        fig.savefig(enhanced_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved enhanced analysis plot: {enhanced_plot_path}")
        
        # Create distribution plots
        _create_distribution_plots(df, fig_dir, scenario_name)
        
    except Exception as e:
        print(f"Warning: Failed to create enhanced analysis plots: {e}")


def _create_distribution_plots(df: pd.DataFrame, fig_dir: str, scenario_name: str):
    """Create distribution plots showing variation across episodes."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create violin plots for key metrics by shift type
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Distribution Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        
        metrics = ['num_los_events', 'min_separation_nm', 'missed_conflict', 'ghost_conflict']
        titles = ['LOS Events Distribution', 'Min Separation Distribution', 
                 'Missed Conflict Rate Distribution', 'Ghost Conflict Rate Distribution']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in df.columns:
                ax = axes[i//2, i%2]
                
                # Create violin plot by shift type
                unique_types = df['shift_type'].unique()
                for j, shift_type in enumerate(unique_types):
                    subset = df[df['shift_type'] == shift_type][metric].dropna()
                    if not subset.empty:
                        parts = ax.violinplot([subset], positions=[j], widths=0.7)
                        # Color the violin plots
                        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                        for pc in parts['bodies']:
                            pc.set_facecolor(colors[j % len(colors)])
                            pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(unique_types)))
                ax.set_xticklabels(unique_types, rotation=45)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save distribution plots
        dist_plot_path = os.path.join(fig_dir, "distribution_analysis.png")
        fig.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved distribution analysis: {dist_plot_path}")
        
    except Exception as e:
        print(f"Warning: Failed to create distribution plots: {e}")


def create_comprehensive_report(results_dir: str, scenario_name: str, output_name: str = "report.html"):
    """
    Create a comprehensive HTML report linking all visualizations.
    
    Args:
        results_dir: Directory containing run results
        scenario_name: Name of the scenario
        output_name: Name of the output HTML file
    """
    try:
        from datetime import datetime
        
        report_path = os.path.join(results_dir, output_name)
        
        # Collect visualization files
        viz_dir = os.path.join(results_dir, "viz")
        shifts_dir = os.path.join(results_dir, "shifts")
        
        # Find PNG files in viz directory
        png_files = []
        if os.path.exists(viz_dir):
            for f in os.listdir(viz_dir):
                if f.endswith('.png'):
                    png_files.append(f)
        
        # Find episode visualization directories
        episode_viz = []
        if os.path.exists(shifts_dir):
            for shift_name in os.listdir(shifts_dir):
                shift_path = os.path.join(shifts_dir, shift_name)
                if os.path.isdir(shift_path):
                    viz_path = os.path.join(shift_path, "viz")
                    if os.path.exists(viz_path):
                        html_files = [f for f in os.listdir(viz_path) if f.endswith('.html')]
                        if html_files:
                            episode_viz.append((shift_name, html_files))
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Visualization Report - {scenario_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .viz-item {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .viz-item img {{ max-width: 100%; height: auto; }}
        .episode-list {{ max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }}
        .episode-item {{ margin: 5px 0; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Visualization Report: {scenario_name}</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Run-Level Visualizations</h2>
        <div class="viz-grid">
"""
        
        # Add PNG visualizations
        for png_file in sorted(png_files):
            png_path = os.path.join("viz", png_file)
            title = png_file.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
            <div class="viz-item">
                <h3>{title}</h3>
                <img src="{png_path}" alt="{title}">
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Episode-Level Visualizations</h2>
        <div class="episode-list">
"""
        
        # Add episode visualizations
        for shift_name, html_files in sorted(episode_viz):
            html_content += f"""
            <div class="episode-item">
                <h3>{shift_name}</h3>
                <ul>
"""
            for html_file in sorted(html_files):
                file_path = os.path.join("shifts", shift_name, "viz", html_file)
                file_title = html_file.replace('.html', '').replace('_', ' ').title()
                html_content += f'<li><a href="{file_path}" target="_blank">{file_title}</a></li>\n'
            
            html_content += "</ul></div>\n"
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Data Files</h2>
        <ul>
            <li><a href="targeted_shift_test_summary.csv">Main Summary CSV</a></li>
            <li><a href="analysis/">Analysis Directory</a></li>
            <li><a href="shifts/">Episode Data Directory</a></li>
        </ul>
    </div>
    
</body>
</html>
"""
        
        # Write HTML file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved: {report_path}")
        
    except Exception as e:
        print(f"Error creating comprehensive report: {e}")