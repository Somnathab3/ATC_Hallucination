"""
Module Name: viz_plotly.py
Description: Interactive temporal analysis and animation using Plotly.
Author: Som
Date: 2025-10-04

Provides interactive visualizations including time series analysis, animated trajectories,
hallucination dashboards, and performance degradation analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def time_series_panel(df, title="Temporal KPIs"):
    """
    Generate multi-metric time series visualization with dual y-axes.
    
    Args:
        df: DataFrame with sim_time_s, min_separation_nm, and agent columns
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive time series with hover data
    """
    g = df.sort_values("sim_time_s")
    fig = go.Figure()
    
    # Add minimum separation traces for each agent
    for aid, gi in g.groupby("agent_id"):
        if "min_separation_nm" in gi.columns:
            fig.add_trace(go.Scatter(
                x=gi["sim_time_s"], 
                y=gi["min_separation_nm"],
                mode="lines", 
                name=f"minSep {aid}",
                line=dict(width=2)
            ))
        
        # Add DCPA if available
        if "tcpa_s" in gi.columns and "dcpa_nm" in gi.columns:
            fig.add_trace(go.Scatter(
                x=gi["sim_time_s"], 
                y=gi["dcpa_nm"],
                mode="lines", 
                name=f"DCPA {aid}", 
                line=dict(dash="dot", width=2)
            ))
    
    # Overplot alert state on secondary y-axis
    if "predicted_alert" in g.columns:
        fig.add_trace(go.Scatter(
            x=g["sim_time_s"], 
            y=(g["predicted_alert"] * 1.0),
            mode="lines", 
            name="Alert", 
            yaxis="y2", 
            line=dict(color="red", width=3)
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying='y', 
                side='right', 
                title='Alert (0/1)',
                range=[-0.1, 1.1]
            )
        )
    
    # Add 5 NM safety threshold line
    fig.add_hline(y=5.0, line_dash="dash", line_color="red", 
                  annotation_text="5 NM Safety Threshold")
    
    fig.update_layout(
        title=title, 
        xaxis_title="Time (s)", 
        yaxis_title="Distance (NM)", 
        template="plotly_white",
        height=500,
        legend=dict(x=1.05, y=1)
    )
    return fig

def animated_geo(df, title="Animated tracks (Plotly)"):
    """
    Create animated geographic plot of aircraft trajectories.
    
    Args:
        df: DataFrame with geographic and temporal data
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Animated map figure
    """
    # Sort data for proper animation
    df_sorted = df.sort_values(["step_idx", "agent_id"])
    
    # Use scatter_mapbox for geographic animation
    fig = px.scatter_mapbox(
        df_sorted,
        lat="lat_deg", 
        lon="lon_deg", 
        color="agent_id",
        size="tas_kt",  # Size by true airspeed if available
        animation_frame="step_idx", 
        hover_data=[
            "sim_time_s", 
            "min_separation_nm", 
            "conflict_flag", 
            "predicted_alert",
            "hdg_deg",
            "tas_kt"
        ],
        zoom=6, 
        height=600,
        title=title
    )
    
    fig.update_layout(
        mapbox_style="open-street-map", 
        legend_title_text="Agent",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    # Customize animation
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 100
    
    return fig

def conflict_evolution_plot(df, title="Conflict Evolution Over Time"):
    """
    Show how conflicts evolve over time with separation distance.
    
    Args:
        df: DataFrame with conflict data
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Conflict evolution figure
    """
    fig = go.Figure()
    
    # Group by agent pairs to show separation
    agents = df["agent_id"].unique()
    
    for i, aid1 in enumerate(agents):
        for aid2 in agents[i+1:]:
            # Calculate pairwise separation (simplified - would need proper implementation)
            pair_name = f"{aid1}-{aid2}"
            
            # For now, use min_separation_nm as proxy
            agent_data = df[df["agent_id"].isin([aid1, aid2])]
            if not agent_data.empty:
                fig.add_trace(go.Scatter(
                    x=agent_data["sim_time_s"],
                    y=agent_data["min_separation_nm"],
                    mode="lines+markers",
                    name=pair_name,
                    line=dict(width=2)
                ))
    
    # Add safety threshold
    fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                  annotation_text="5 NM Safety Threshold")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Separation Distance (NM)",
        template="plotly_white",
        height=400
    )
    
    return fig

def alert_timeline(df, title="Alert Timeline"):
    """
    Create timeline showing when alerts were triggered.
    
    Args:
        df: DataFrame with alert data
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Alert timeline figure
    """
    fig = go.Figure()
    
    for aid, agent_data in df.groupby("agent_id"):
        # Show alert periods
        alert_times = agent_data[agent_data["predicted_alert"] == 1]["sim_time_s"]
        
        if not alert_times.empty:
            fig.add_trace(go.Scatter(
                x=alert_times,
                y=[aid] * len(alert_times),
                mode="markers",
                marker=dict(size=10, symbol="square"),
                name=f"Alerts {aid}",
                showlegend=True
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Agent",
        template="plotly_white",
        height=300
    )
    
    return fig

def hallucination_dashboard(df, title="Hallucination Analysis Dashboard"):
    """
    Create comprehensive dashboard showing hallucination metrics.
    
    Args:
        df: DataFrame with hallucination data
        title: Dashboard title
        
    Returns:
        plotly.graph_objects.Figure: Dashboard figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Confusion Matrix Over Time",
            "False Positive Rate by Agent", 
            "Alert Accuracy Timeline",
            "Conflict Detection Performance"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Calculate metrics
    df['fp_rate'] = df.get('fp', 0) / (df.get('fp', 0) + df.get('tn', 1))
    df['fn_rate'] = df.get('fn', 0) / (df.get('fn', 0) + df.get('tp', 1))
    df['accuracy'] = (df.get('tp', 0) + df.get('tn', 0)) / (
        df.get('tp', 0) + df.get('tn', 0) + df.get('fp', 0) + df.get('fn', 0) + 1e-8
    )
    
    # Plot 1: Confusion matrix over time
    for metric, color in [('fp_rate', 'red'), ('fn_rate', 'orange')]:
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["sim_time_s"],
                    y=df[metric],
                    mode="lines",
                    name=metric.upper(),
                    line=dict(color=color)
                ),
                row=1, col=1
            )
    
    # Plot 2: FP rate by agent
    if 'fp_rate' in df.columns:
        fp_by_agent = df.groupby('agent_id')['fp_rate'].mean()
        fig.add_trace(
            go.Bar(
                x=fp_by_agent.index,
                y=fp_by_agent.values,
                name="FP Rate",
                marker_color="red"
            ),
            row=1, col=2
        )
    
    # Plot 3: Accuracy timeline
    if 'accuracy' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["sim_time_s"],
                y=df["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="green")
            ),
            row=2, col=1
        )
    
    # Plot 4: Detection performance
    if all(col in df.columns for col in ['min_separation_nm', 'predicted_alert']):
        fig.add_trace(
            go.Scatter(
                x=df["min_separation_nm"],
                y=df["predicted_alert"],
                mode="markers",
                name="Alert vs Separation",
                marker=dict(color=df.get('conflict_flag', 0), colorscale='RdYlBu')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text=title,
        height=800,
        template="plotly_white",
        showlegend=True
    )
    
    return fig

def performance_degradation_plot(summary_df, metric_col="fn_rate", 
                                shift_col="shift_magnitude", 
                                group_col="shift_type"):
    """
    Plot performance degradation across shift magnitudes.
    
    Args:
        summary_df: Summary DataFrame with aggregated metrics
        metric_col: Column containing the metric to plot
        shift_col: Column containing shift magnitudes
        group_col: Column for grouping (e.g., shift type)
        
    Returns:
        plotly.graph_objects.Figure: Performance degradation plot
    """
    fig = go.Figure()
    
    for shift_type, group_data in summary_df.groupby(group_col):
        # Aggregate by shift magnitude
        agg_data = group_data.groupby(shift_col)[metric_col].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(go.Scatter(
            x=agg_data[shift_col],
            y=agg_data['mean'],
            mode='lines+markers',
            name=shift_type,
            line=dict(width=3),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=agg_data['std'],
                visible=True
            )
        ))
    
    fig.update_layout(
        title=f"Performance Degradation: {metric_col} vs {shift_col}",
        xaxis_title=shift_col.replace('_', ' ').title(),
        yaxis_title=metric_col.replace('_', ' ').title(),
        template="plotly_white",
        height=500
    )
    
    return fig

def create_interactive_report(baseline_df, shifted_df=None, summary_df=None, 
                            out_html="interactive_report.html"):
    """
    Create comprehensive interactive HTML report.
    
    Args:
        baseline_df: Baseline trajectory data
        shifted_df: Shifted trajectory data (optional)
        summary_df: Summary statistics (optional)
        out_html: Output HTML filename
        
    Returns:
        str: Path to saved HTML file
    """
    from plotly.offline import plot
    from plotly.subplots import make_subplots
    
    # Use shifted data if available
    analysis_df = shifted_df if shifted_df is not None else baseline_df
    
    # Create individual plots
    plots = []
    
    # Time series
    plots.append(time_series_panel(analysis_df, "Key Performance Indicators"))
    
    # Animated map
    plots.append(animated_geo(analysis_df, "Aircraft Trajectory Animation"))
    
    # Hallucination dashboard
    if any(col in analysis_df.columns for col in ['fp', 'fn', 'tp', 'tn']):
        plots.append(hallucination_dashboard(analysis_df, "Hallucination Analysis"))
    
    # Performance degradation if summary available
    if summary_df is not None and len(summary_df) > 0:
        plots.append(performance_degradation_plot(summary_df))
    
    # Save all plots to HTML
    with open(out_html, 'w') as f:
        f.write("<html><head><title>ATC Hallucination Analysis Report</title></head><body>")
        f.write("<h1>Interactive Analysis Report</h1>")
        
        for i, fig in enumerate(plots):
            plot_html = plot(fig, output_type='div', include_plotlyjs=(i==0))
            f.write(plot_html)
        
        f.write("</body></html>")
    
    return out_html