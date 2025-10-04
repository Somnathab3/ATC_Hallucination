"""
Module Name: make_all_figures.py
Description: Master orchestration script for comprehensive analysis figure generation.
Author: Som
Date: 2025-10-04

Central coordination for generating complete sets of academic-quality visualizations
from ATC hallucination detection research data for thesis and publication preparation.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import analysis modules
from analysis.ingest import load_complete_dataset, export_harmonized_data
from analysis.viz_geographic import build_map, create_comparison_map, create_hotspot_map
from analysis.viz_plotly import (time_series_panel, animated_geo, hallucination_dashboard, 
                                performance_degradation_plot, create_interactive_report)
from analysis.viz_matplotlib import create_publication_figure_set
from analysis.analysis_hotspots import analyze_hotspots, identify_conflict_zones
from analysis.analysis_similarity import analyze_trajectory_clustering
from analysis.analysis_safety import calculate_safety_metrics, performance_degradation_analysis

def setup_output_directories(base_dir="figures_output"):
    """
    Create organized output directory structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        dict: Dictionary of output directories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"{base_dir}_{timestamp}"
    
    directories = {
        'root': output_root,
        'maps': os.path.join(output_root, "geographic_maps"),
        'interactive': os.path.join(output_root, "interactive_plots"),
        'static': os.path.join(output_root, "publication_figures"),
        'analysis': os.path.join(output_root, "analysis_results"),
        'data': os.path.join(output_root, "processed_data"),
        'reports': os.path.join(output_root, "reports")
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def generate_geographic_maps(dataset, output_dirs):
    """
    Generate all geographic visualization maps.
    
    Args:
        dataset: Complete dataset dictionary
        output_dirs: Output directory structure
        
    Returns:
        dict: Generated map files
    """
    print("Generating geographic maps...")
    
    trajectories = dataset['trajectories']
    map_files = {}
    
    if trajectories.empty:
        print("Warning: No trajectory data available for maps")
        return map_files
    
    # Group data by scenario and shift type for comparison maps
    scenarios = trajectories.get('scenario', pd.Series()).unique()
    shift_types = trajectories.get('shift_type', pd.Series()).unique()
    
    # Overall trajectory map
    if not trajectories.empty:
        overall_map = os.path.join(output_dirs['maps'], "overview_trajectories.html")
        build_map(trajectories, out_html=overall_map)
        map_files['overview'] = overall_map
    
    # Scenario-specific maps
    for scenario in scenarios:
        if pd.isna(scenario):
            continue
            
        scenario_data = trajectories[trajectories['scenario'] == scenario]
        
        if not scenario_data.empty:
            scenario_map = os.path.join(output_dirs['maps'], f"scenario_{scenario}.html")
            build_map(scenario_data, out_html=scenario_map)
            map_files[f'scenario_{scenario}'] = scenario_map
    
    # Comparison maps (baseline vs shifted)
    baseline_data = trajectories[trajectories.get('shift_magnitude', 0) == 0]
    shifted_data = trajectories[trajectories.get('shift_magnitude', 0) > 0]
    
    if not baseline_data.empty and not shifted_data.empty:
        comparison_map = os.path.join(output_dirs['maps'], "baseline_vs_shifted.html")
        create_comparison_map(baseline_data, shifted_data, "baseline_vs_shifted", output_dirs['maps'])
        map_files['comparison'] = comparison_map
    
    # Hotspot analysis map
    if any(col in trajectories.columns for col in ['fp', 'fn', 'conflict_flag']):
        hotspot_map = os.path.join(output_dirs['maps'], "conflict_hotspots.html")
        create_hotspot_map(trajectories, hotspot_map)
        map_files['hotspots'] = hotspot_map
    
    print(f"Generated {len(map_files)} geographic maps")
    return map_files

def generate_interactive_plots(dataset, output_dirs):
    """
    Generate interactive Plotly visualizations.
    
    Args:
        dataset: Complete dataset dictionary
        output_dirs: Output directory structure
        
    Returns:
        dict: Generated plot files
    """
    print("Generating interactive plots...")
    
    trajectories = dataset['trajectories']
    summary = dataset['summary']
    plot_files = {}
    
    if trajectories.empty:
        print("Warning: No trajectory data available for interactive plots")
        return plot_files
    
    # Time series analysis
    try:
        time_fig = time_series_panel(trajectories, "Key Performance Indicators Over Time")
        time_file = os.path.join(output_dirs['interactive'], "time_series_kpi.html")
        time_fig.write_html(time_file)
        plot_files['time_series'] = time_file
    except Exception as e:
        print(f"Warning: Failed to generate time series plot: {e}")
    
    # Animated geographic plot
    try:
        anim_fig = animated_geo(trajectories, "Aircraft Trajectory Animation")
        anim_file = os.path.join(output_dirs['interactive'], "animated_trajectories.html")
        anim_fig.write_html(anim_file)
        plot_files['animation'] = anim_file
    except Exception as e:
        print(f"Warning: Failed to generate animated plot: {e}")
    
    # Hallucination dashboard
    if any(col in trajectories.columns for col in ['fp', 'fn', 'tp', 'tn']):
        try:
            hall_fig = hallucination_dashboard(trajectories, "Hallucination Analysis Dashboard")
            hall_file = os.path.join(output_dirs['interactive'], "hallucination_dashboard.html")
            hall_fig.write_html(hall_file)
            plot_files['hallucination'] = hall_file
        except Exception as e:
            print(f"Warning: Failed to generate hallucination dashboard: {e}")
    
    # Performance degradation (if summary data available)
    if not summary.empty and 'shift_magnitude' in summary.columns:
        try:
            for metric in ['fn_rate', 'fp_rate', 'resolution_failure_rate']:
                if metric in summary.columns:
                    perf_fig = performance_degradation_plot(summary, metric)
                    perf_file = os.path.join(output_dirs['interactive'], f"degradation_{metric}.html")
                    perf_fig.write_html(perf_file)
                    plot_files[f'degradation_{metric}'] = perf_file
                    break  # Generate one example
        except Exception as e:
            print(f"Warning: Failed to generate performance plot: {e}")
    
    # Comprehensive interactive report
    try:
        report_file = os.path.join(output_dirs['interactive'], "comprehensive_report.html")
        create_interactive_report(trajectories, None, summary, report_file)
        plot_files['comprehensive_report'] = report_file
    except Exception as e:
        print(f"Warning: Failed to generate comprehensive report: {e}")
    
    print(f"Generated {len(plot_files)} interactive plots")
    return plot_files

def generate_publication_figures(dataset, output_dirs):
    """
    Generate publication-quality static figures.
    
    Args:
        dataset: Complete dataset dictionary
        output_dirs: Output directory structure
        
    Returns:
        dict: Generated figure files
    """
    print("Generating publication figures...")
    
    trajectories = dataset['trajectories']
    summary = dataset['summary']
    
    if trajectories.empty and summary.empty:
        print("Warning: No data available for publication figures")
        return {}
    
    try:
        # Use summary data if available, otherwise use trajectory data
        summary_data = summary if not summary.empty else trajectories
        trajectory_data = trajectories if not trajectories.empty else summary
        
        figure_files = create_publication_figure_set(
            summary_data, 
            trajectory_data, 
            output_dirs['static']
        )
        
        print(f"Generated {len(figure_files)} publication figures")
        return figure_files
        
    except Exception as e:
        print(f"Warning: Failed to generate publication figures: {e}")
        return {}

def perform_comprehensive_analysis(dataset, output_dirs):
    """
    Perform comprehensive analysis and save results.
    
    Args:
        dataset: Complete dataset dictionary
        output_dirs: Output directory structure
        
    Returns:
        dict: Analysis results
    """
    print("Performing comprehensive analysis...")
    
    trajectories = dataset['trajectories']
    analysis_results = {}
    
    if trajectories.empty:
        print("Warning: No trajectory data for analysis")
        return analysis_results
    
    # Hotspot analysis
    try:
        hotspots = analyze_hotspots(trajectories, ['fp', 'fn'], eps_nm=3.0, min_samples=5)
        analysis_results['hotspots'] = hotspots
        
        # Save hotspot results
        hotspot_file = os.path.join(output_dirs['analysis'], "hotspot_analysis.json")
        with open(hotspot_file, 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            hotspots_serializable = {}
            for key, value in hotspots.items():
                if isinstance(value, dict):
                    hotspots_serializable[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'to_dict'):
                            hotspots_serializable[key][k] = v.to_dict()
                        else:
                            hotspots_serializable[key][k] = v
                else:
                    hotspots_serializable[key] = value
            json.dump(hotspots_serializable, f, indent=2, default=str)
        
    except Exception as e:
        print(f"Warning: Hotspot analysis failed: {e}")
    
    # Conflict zone analysis
    try:
        conflict_zones = identify_conflict_zones(trajectories, conflict_threshold_nm=5.0)
        analysis_results['conflict_zones'] = conflict_zones
        
        zones_file = os.path.join(output_dirs['analysis'], "conflict_zones.json")
        with open(zones_file, 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            zones_serializable = {}
            for key, value in conflict_zones.items():
                if hasattr(value, 'to_dict'):
                    zones_serializable[key] = value.to_dict()
                else:
                    zones_serializable[key] = value
            json.dump(zones_serializable, f, indent=2, default=str)
        
    except Exception as e:
        print(f"Warning: Conflict zone analysis failed: {e}")
    
    # Safety metrics
    try:
        safety_metrics = calculate_safety_metrics(trajectories)
        analysis_results['safety_metrics'] = safety_metrics
        
        safety_file = os.path.join(output_dirs['analysis'], "safety_metrics.csv")
        safety_metrics.to_csv(safety_file, index=False)
        
    except Exception as e:
        print(f"Warning: Safety analysis failed: {e}")
    
    # Trajectory similarity (if we have baseline and shifted data)
    try:
        baseline_data = trajectories[trajectories.get('shift_magnitude', 0) == 0]
        shifted_data = trajectories[trajectories.get('shift_magnitude', 0) > 0]
        
        if not baseline_data.empty and not shifted_data.empty:
            similarity_analysis = analyze_trajectory_clustering(baseline_data, shifted_data)
            analysis_results['trajectory_similarity'] = similarity_analysis
            
            similarity_file = os.path.join(output_dirs['analysis'], "trajectory_similarity.json")
            with open(similarity_file, 'w') as f:
                json.dump(similarity_analysis, f, indent=2, default=str)
    
    except Exception as e:
        print(f"Warning: Trajectory similarity analysis failed: {e}")
    
    print(f"Completed {len(analysis_results)} analysis components")
    return analysis_results

def generate_summary_report(dataset, output_dirs, map_files, plot_files, figure_files, analysis_results):
    """
    Generate a comprehensive summary report.
    
    Args:
        dataset: Complete dataset dictionary
        output_dirs: Output directory structure
        map_files: Generated map files
        plot_files: Generated plot files
        figure_files: Generated figure files
        analysis_results: Analysis results
        
    Returns:
        str: Path to summary report
    """
    print("Generating summary report...")
    
    report_file = os.path.join(output_dirs['reports'], "analysis_summary.html")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATC Hallucination Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
            .file-list {{ background-color: #ffffff; padding: 15px; border-radius: 5px; }}
            .file-list ul {{ margin: 10px 0; }}
            .metadata {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>ATC Hallucination Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="metadata">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Trajectory Records</td><td>{dataset['metadata']['n_trajectory_records']}</td></tr>
                    <tr><td>Hallucination Records</td><td>{dataset['metadata']['n_hallucination_records']}</td></tr>
                    <tr><td>Summary Records</td><td>{dataset['metadata']['n_summary_records']}</td></tr>
                    <tr><td>Trajectory Columns</td><td>{len(dataset['metadata']['trajectory_columns'])}</td></tr>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>Generated Visualizations</h2>
            <h3>Geographic Maps ({len(map_files)} files)</h3>
            <div class="file-list">
                <ul>
    """
    
    for name, path in map_files.items():
        rel_path = os.path.relpath(path, output_dirs['reports'])
        html_content += f"<li><a href='{rel_path}'>{name.replace('_', ' ').title()}</a></li>"
    
    html_content += f"""
                </ul>
            </div>
            
            <h3>Interactive Plots ({len(plot_files)} files)</h3>
            <div class="file-list">
                <ul>
    """
    
    for name, path in plot_files.items():
        rel_path = os.path.relpath(path, output_dirs['reports'])
        html_content += f"<li><a href='{rel_path}'>{name.replace('_', ' ').title()}</a></li>"
    
    html_content += f"""
                </ul>
            </div>
            
            <h3>Publication Figures ({len(figure_files)} files)</h3>
            <div class="file-list">
                <ul>
    """
    
    for name, path in figure_files.items():
        rel_path = os.path.relpath(path, output_dirs['reports'])
        html_content += f"<li><a href='{rel_path}'>{name.replace('_', ' ').title()}</a></li>"
    
    html_content += f"""
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
            <p>Comprehensive analysis performed on {len(analysis_results)} components:</p>
            <ul>
    """
    
    for component in analysis_results.keys():
        html_content += f"<li>{component.replace('_', ' ').title()}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Directory Structure</h2>
            <pre>
    """
    
    for name, path in output_dirs.items():
        rel_path = os.path.relpath(path, output_dirs['root'])
        html_content += f"{name}: {rel_path}/\n"
    
    html_content += """
            </pre>
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"Summary report generated: {report_file}")
    return report_file

def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Generate all ATC hallucination analysis figures")
    parser.add_argument("--results-dir", default="results", help="Results directory path")
    parser.add_argument("--output-dir", default="figures_output", help="Output directory base name")
    parser.add_argument("--skip-maps", action="store_true", help="Skip geographic map generation")
    parser.add_argument("--skip-interactive", action="store_true", help="Skip interactive plot generation")
    parser.add_argument("--skip-static", action="store_true", help="Skip static figure generation")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip comprehensive analysis")
    
    args = parser.parse_args()
    
    print("=== ATC Hallucination Analysis Figure Generation ===")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup output directories
    output_dirs = setup_output_directories(args.output_dir)
    print(f"Output root: {output_dirs['root']}")
    
    # Load complete dataset
    print("\n=== Loading Dataset ===")
    dataset = load_complete_dataset(args.results_dir)
    
    if dataset['metadata']['n_trajectory_records'] == 0:
        print("Warning: No trajectory data found. Limited analysis will be performed.")
    
    # Export harmonized data
    export_harmonized_data(dataset, output_dirs['data'])
    
    # Generate visualizations
    map_files = {}
    plot_files = {}
    figure_files = {}
    analysis_results = {}
    
    if not args.skip_maps:
        print("\n=== Generating Geographic Maps ===")
        map_files = generate_geographic_maps(dataset, output_dirs)
    
    if not args.skip_interactive:
        print("\n=== Generating Interactive Plots ===")
        plot_files = generate_interactive_plots(dataset, output_dirs)
    
    if not args.skip_static:
        print("\n=== Generating Publication Figures ===")
        figure_files = generate_publication_figures(dataset, output_dirs)
    
    if not args.skip_analysis:
        print("\n=== Performing Comprehensive Analysis ===")
        analysis_results = perform_comprehensive_analysis(dataset, output_dirs)
    
    # Generate summary report
    print("\n=== Generating Summary Report ===")
    report_file = generate_summary_report(
        dataset, output_dirs, map_files, plot_files, figure_files, analysis_results
    )
    
    # Print summary
    print("\n=== Generation Complete ===")
    print(f"Total files generated:")
    print(f"  Geographic maps: {len(map_files)}")
    print(f"  Interactive plots: {len(plot_files)}")
    print(f"  Publication figures: {len(figure_files)}")
    print(f"  Analysis components: {len(analysis_results)}")
    print(f"\nAll outputs saved to: {output_dirs['root']}")
    print(f"Summary report: {report_file}")

if __name__ == "__main__":
    # Add pandas import at module level to fix the issue
    import pandas as pd
    main()