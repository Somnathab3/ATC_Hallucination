"""
Module Name: viz_matplotlib.py
Description: Publication-quality static visualization using Matplotlib.
Author: Som
Date: 2025-10-04

Generates high-quality static figures suitable for academic publications with
bootstrap confidence intervals and professional styling.
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def bootstrap_ci(a, n_boot=1000, agg=np.mean, alpha=0.05, rng=None):
    """
    Compute bootstrap confidence intervals for uncertainty quantification.
    
    Args:
        a: Array of sample values
        n_boot: Number of bootstrap samples (default: 1000)
        agg: Aggregation function (default: np.mean)
        alpha: Significance level (default: 0.05 for 95% CI)
        rng: Random number generator for reproducibility
        
    Returns:
        tuple: (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42) if rng is None else rng
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    
    if a.size == 0:
        return np.nan, np.nan, np.nan
        
    bs = [agg(rng.choice(a, size=a.size, replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [100*alpha/2, 100*(1-alpha/2)])
    
    return float(np.mean(bs)), float(lo), float(hi)

def plot_degradation_curves(df, metric_col, shift_col="shift_magnitude", 
                          group_col="shift_type", title=None, figsize=(10, 6)):
    """
    Plot performance degradation curves with confidence intervals.
    
    Args:
        df: DataFrame with performance data
        metric_col: Column containing the metric to plot
        shift_col: Column containing shift magnitudes
        group_col: Column for grouping curves
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df[group_col].unique())))
    
    for i, (k, g) in enumerate(df.groupby(group_col)):
        xm = sorted(g[shift_col].unique())
        ym, lo, hi = [], [], []
        
        for x in xm:
            vals = g.loc[g[shift_col] == x, metric_col].values
            m, l, h = bootstrap_ci(vals)
            ym.append(m)
            lo.append(l) 
            hi.append(h)
        
        color = colors[i]
        ax.plot(xm, ym, label=k, color=color, linewidth=2.5, marker='o', markersize=6)
        ax.fill_between(xm, lo, hi, alpha=0.3, color=color)
    
    ax.set_xlabel(shift_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

def heatmap_agent_vulnerability(df, metric_col, agents=("A0", "A1", "A2", "A3"), 
                               magnitudes=None, cmap="RdYlBu_r", figsize=(10, 6)):
    """
    Create heatmap showing agent vulnerability across shift magnitudes.
    
    Args:
        df: DataFrame with performance data
        metric_col: Column containing the metric
        agents: List of agent IDs
        magnitudes: List of shift magnitudes (auto-detected if None)
        cmap: Colormap name
        figsize: Figure size tuple
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if magnitudes is None:
        magnitudes = sorted(df["shift_magnitude"].unique())
    
    # Create vulnerability matrix
    M = np.zeros((len(agents), len(magnitudes)))
    
    for i, a in enumerate(agents):
        for j, m in enumerate(magnitudes):
            vals = df[(df["agent_id"] == a) & (df["shift_magnitude"] == m)][metric_col].values
            M[i, j] = np.nanmean(vals) if len(vals) > 0 else np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
    
    # Set ticks and labels
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"{m:.1f}" for m in magnitudes])
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_col.replace('_', ' ').title(), fontsize=12)
    
    # Add value annotations
    for i in range(len(agents)):
        for j in range(len(magnitudes)):
            if not np.isnan(M[i, j]):
                text = ax.text(j, i, f'{M[i, j]:.3f}', 
                             ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel("Shift Magnitude", fontsize=12)
    ax.set_ylabel("Agent", fontsize=12)
    ax.set_title(f"Agent Vulnerability: {metric_col.replace('_', ' ').title()}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def confusion_matrix_evolution(df, shift_bins=None, figsize=(12, 8)):
    """
    Show evolution of confusion matrix across shift magnitudes.
    
    Args:
        df: DataFrame with confusion matrix data (fp, fn, tp, tn columns)
        shift_bins: Bins for shift magnitudes
        figsize: Figure size tuple
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if shift_bins is None:
        shift_bins = np.linspace(df["shift_magnitude"].min(), 
                               df["shift_magnitude"].max(), 6)
    
    # Bin the data
    df['shift_bin'] = pd.cut(df['shift_magnitude'], bins=shift_bins, include_lowest=True)
    
    # Aggregate confusion matrix by bins
    cm_cols = ['tp', 'tn', 'fp', 'fn']
    available_cols = [col for col in cm_cols if col in df.columns]
    
    if not available_cols:
        print("Warning: No confusion matrix columns found")
        return plt.figure()
    
    grouped = df.groupby('shift_bin')[available_cols].sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Stacked bar chart
    bottom = np.zeros(len(grouped))
    colors = ['green', 'lightgreen', 'orange', 'red']
    labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    
    for i, col in enumerate(available_cols):
        if col in grouped.columns:
            ax1.bar(range(len(grouped)), grouped[col], bottom=bottom, 
                   color=colors[cm_cols.index(col)], label=labels[cm_cols.index(col)])
            bottom += grouped[col]
    
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels([f"{interval.left:.1f}-{interval.right:.1f}" 
                        for interval in grouped.index], rotation=45)
    ax1.set_xlabel("Shift Magnitude Bins")
    ax1.set_ylabel("Count")
    ax1.set_title("Confusion Matrix Evolution (Stacked)")
    ax1.legend()
    
    # Heatmap of rates
    if len(available_cols) >= 4:
        # Calculate rates
        total = grouped.sum(axis=1)
        rates = grouped.div(total, axis=0)
        
        im = ax2.imshow(rates.T, aspect='auto', cmap='RdYlBu_r')
        ax2.set_xticks(range(len(grouped)))
        ax2.set_xticklabels([f"{interval.left:.1f}-{interval.right:.1f}" 
                            for interval in grouped.index], rotation=45)
        ax2.set_yticks(range(len(available_cols)))
        ax2.set_yticklabels([col.upper() for col in available_cols])
        ax2.set_title("Confusion Matrix Rates")
        
        plt.colorbar(im, ax=ax2, label="Rate")
    
    plt.tight_layout()
    return fig

def oscillation_patterns(df, agents=None, figsize=(10, 6)):
    """
    Plot action oscillation patterns per agent.
    
    Args:
        df: DataFrame with action data
        agents: List of agents to analyze (auto-detected if None)
        figsize: Figure size tuple
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if agents is None:
        agents = sorted(df["agent_id"].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    oscillation_data = []
    
    for agent in agents:
        agent_df = df[df["agent_id"] == agent].sort_values("step_idx")
        
        # Calculate action reversals (simplified)
        if "action_hdg_delta_deg" in agent_df.columns:
            hdg_changes = agent_df["action_hdg_delta_deg"].diff()
            # Count sign changes as oscillations
            oscillations = (hdg_changes[1:] * hdg_changes[:-1].values < 0).sum()
            oscillation_data.append(oscillations)
        else:
            oscillation_data.append(0)
    
    # Create bar chart
    bars = ax.bar(agents, oscillation_data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, value in zip(bars, oscillation_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Action Reversals", fontsize=12)
    ax.set_title("Action Oscillation Patterns by Agent", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def safety_margin_distribution(df, figsize=(12, 8)):
    """
    Plot distribution of safety margins (minimum separation).
    
    Args:
        df: DataFrame with separation data
        figsize: Figure size tuple
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Overall distribution
    if "min_separation_nm" in df.columns:
        ax1.hist(df["min_separation_nm"].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='5 NM Safety Threshold')
        ax1.set_xlabel("Minimum Separation (NM)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Minimum Separation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # By agent
    if "agent_id" in df.columns and "min_separation_nm" in df.columns:
        for i, (agent, agent_data) in enumerate(df.groupby("agent_id")):
            ax2.hist(agent_data["min_separation_nm"].dropna(), bins=20, alpha=0.6, 
                    label=f"Agent {agent}")
        ax2.axvline(x=5.0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel("Minimum Separation (NM)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Separation Distribution by Agent")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Time series of violations
    if "sim_time_s" in df.columns and "min_separation_nm" in df.columns:
        violations = df[df["min_separation_nm"] < 5.0]
        if not violations.empty:
            ax3.scatter(violations["sim_time_s"], violations["min_separation_nm"], 
                       c='red', alpha=0.6, s=20)
            ax3.axhline(y=5.0, color='red', linestyle='--', linewidth=2)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Separation (NM)")
            ax3.set_title("Safety Violations Over Time")
            ax3.grid(True, alpha=0.3)
    
    # Conflict duration
    if "conflict_flag" in df.columns and "sim_time_s" in df.columns:
        conflicts = df[df["conflict_flag"] == 1]
        if not conflicts.empty:
            conflict_durations = []
            for episode in conflicts["episode_id"].unique():
                ep_conflicts = conflicts[conflicts["episode_id"] == episode]
                if len(ep_conflicts) > 1:
                    duration = ep_conflicts["sim_time_s"].max() - ep_conflicts["sim_time_s"].min()
                    conflict_durations.append(duration)
            
            if conflict_durations:
                ax4.hist(conflict_durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
                ax4.set_xlabel("Conflict Duration (s)")
                ax4.set_ylabel("Frequency")
                ax4.set_title("Distribution of Conflict Durations")
                ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_publication_figure_set(summary_df, trajectory_df, out_dir="figures"):
    """
    Create complete set of publication-ready figures.
    
    Args:
        summary_df: Summary statistics DataFrame
        trajectory_df: Detailed trajectory DataFrame
        out_dir: Output directory for figures
        
    Returns:
        dict: Mapping of figure names to file paths
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    figures = {}
    
    # Performance degradation curves
    if "fn_rate" in summary_df.columns:
        fig1 = plot_degradation_curves(summary_df, "fn_rate", 
                                     title="False Negative Rate vs Shift Magnitude")
        fig1.savefig(os.path.join(out_dir, "degradation_fn_rate.png"), dpi=300, bbox_inches='tight')
        figures["degradation_fn"] = os.path.join(out_dir, "degradation_fn_rate.png")
        plt.close(fig1)
    
    # Agent vulnerability heatmap
    if "resolution_failure_rate" in summary_df.columns:
        fig2 = heatmap_agent_vulnerability(summary_df, "resolution_failure_rate")
        fig2.savefig(os.path.join(out_dir, "agent_vulnerability.png"), dpi=300, bbox_inches='tight')
        figures["vulnerability"] = os.path.join(out_dir, "agent_vulnerability.png")
        plt.close(fig2)
    
    # Confusion matrix evolution
    if any(col in trajectory_df.columns for col in ['fp', 'fn', 'tp', 'tn']):
        fig3 = confusion_matrix_evolution(trajectory_df)
        fig3.savefig(os.path.join(out_dir, "confusion_evolution.png"), dpi=300, bbox_inches='tight')
        figures["confusion"] = os.path.join(out_dir, "confusion_evolution.png")
        plt.close(fig3)
    
    # Oscillation patterns
    fig4 = oscillation_patterns(trajectory_df)
    fig4.savefig(os.path.join(out_dir, "oscillation_patterns.png"), dpi=300, bbox_inches='tight')
    figures["oscillation"] = os.path.join(out_dir, "oscillation_patterns.png")
    plt.close(fig4)
    
    # Safety margin distribution
    fig5 = safety_margin_distribution(trajectory_df)
    fig5.savefig(os.path.join(out_dir, "safety_margins.png"), dpi=300, bbox_inches='tight')
    figures["safety"] = os.path.join(out_dir, "safety_margins.png")
    plt.close(fig5)
    
    return figures