"""
Visualize confusion matrix distribution across shift types.
Shows how TP, FP, FN, TN vary with different perturbations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def visualize_confusion_distribution(summary_csv: str, output_dir: str = "confusion_matrix_analysis"):
    """
    Create visualizations showing confusion matrix patterns across shifts.
    
    Args:
        summary_csv: Path to targeted_shift_test_summary.csv
        output_dir: Directory to save visualization outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(summary_csv)
    
    print(f"Loaded {len(df)} test cases from {summary_csv}")
    print(f"Shift types: {df['shift_type'].unique().tolist()}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Confusion Matrix Distribution Across Shift Types', fontsize=16, fontweight='bold')
    
    # 1. Event counts by shift type
    ax = axes[0, 0]
    shift_metrics = df.groupby('shift_type')[['tp', 'fp', 'fn']].sum()
    shift_metrics.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_title('Event Counts by Shift Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Event Count')
    ax.legend(['True Positives', 'False Positives', 'False Negatives'])
    ax.grid(True, alpha=0.3)
    
    # 2. TN distribution (timesteps)
    ax = axes[0, 1]
    shift_tn = df.groupby('shift_type')['tn'].mean()
    shift_tn.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Average True Negatives (Timesteps) by Shift Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Average TN Timesteps')
    ax.axhline(y=60, color='gray', linestyle='--', label='Typical baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision and Recall by shift type
    ax = axes[1, 0]
    shift_perf = df.groupby('shift_type')[['precision', 'recall']].mean()
    shift_perf.plot(kind='bar', ax=ax, color=['purple', 'teal'])
    ax.set_title('Detection Performance by Shift Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.1)
    ax.legend(['Precision', 'Recall'])
    ax.grid(True, alpha=0.3)
    
    # 4. Min separation distribution
    ax = axes[1, 1]
    shift_sep = df.groupby('shift_type')['min_separation_nm'].agg(['min', 'mean', 'max'])
    shift_sep.plot(kind='bar', ax=ax, color=['red', 'orange', 'yellow'])
    ax.set_title('Minimum Separation by Shift Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Separation (NM)')
    ax.axhline(y=5.0, color='red', linestyle='--', label='LoS threshold (5 NM)')
    ax.axhline(y=6.0, color='orange', linestyle='--', label='Threat gate (6 NM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_overview.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/confusion_matrix_overview.png")
    
    # Create detailed heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot data for heatmap
    heatmap_data = df.groupby(['shift_type', 'target_agent']).agg({
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum',
        'min_separation_nm': 'min'
    }).reset_index()
    
    # Create pivot for each metric
    for metric in ['tp', 'fp', 'fn']:
        pivot = heatmap_data.pivot_table(
            index='shift_type',
            columns='target_agent',
            values=metric,
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Event Count'})
        ax.set_title(f'{metric.upper()} Events by Shift Type and Target Agent', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Agent')
        ax.set_ylabel('Shift Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/{metric}_heatmap.png")
        plt.close()
    
    # Create scatter plot: min_separation vs confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrix Events vs Minimum Separation', fontsize=14, fontweight='bold')
    
    for idx, (metric, color, label) in enumerate([
        ('tp', 'green', 'True Positives'),
        ('fp', 'orange', 'False Positives'),
        ('fn', 'red', 'False Negatives')
    ]):
        ax = axes[idx]
        for shift_type in df['shift_type'].unique():
            subset = df[df['shift_type'] == shift_type]
            ax.scatter(subset['min_separation_nm'], subset[metric], alpha=0.6, label=shift_type, s=50)
        
        ax.set_xlabel('Minimum Separation (NM)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.axvline(x=5.0, color='red', linestyle='--', alpha=0.5, label='LoS threshold')
        ax.axvline(x=6.0, color='orange', linestyle='--', alpha=0.5, label='Threat gate')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'separation_vs_confusion.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/separation_vs_confusion.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CONFUSION MATRIX SUMMARY STATISTICS")
    print("="*80)
    
    print("\n1. Event Counts by Shift Type:")
    print(df.groupby('shift_type')[['tp', 'fp', 'fn', 'tn']].agg({
        'tp': ['sum', 'mean'],
        'fp': ['sum', 'mean'],
        'fn': ['sum', 'mean'],
        'tn': ['mean', 'std']
    }).round(2))
    
    print("\n2. Detection Performance by Shift Type:")
    print(df.groupby('shift_type')[['precision', 'recall', 'f1_score']].agg(['mean', 'std']).round(3))
    
    print("\n3. Separation Statistics by Shift Type:")
    print(df.groupby('shift_type')['min_separation_nm'].agg(['min', 'mean', 'max']).round(2))
    
    print("\n4. Cases with TP > 0:")
    tp_cases = df[df['tp'] > 0][['test_id', 'shift_type', 'shift_value', 'tp', 'fp', 'fn', 'min_separation_nm']]
    print(tp_cases.to_string(index=False))
    
    print("\n5. Cases with FP > 0:")
    fp_cases = df[df['fp'] > 0][['test_id', 'shift_type', 'shift_value', 'tp', 'fp', 'fn', 'min_separation_nm']].head(15)
    print(fp_cases.to_string(index=False))
    
    print("\n6. Overall Statistics:")
    print(f"Total test cases: {len(df)}")
    print(f"Cases with TP > 0: {len(df[df['tp'] > 0])} ({100*len(df[df['tp'] > 0])/len(df):.1f}%)")
    print(f"Cases with FP > 0: {len(df[df['fp'] > 0])} ({100*len(df[df['fp'] > 0])/len(df):.1f}%)")
    print(f"Cases with FN > 0: {len(df[df['fn'] > 0])} ({100*len(df[df['fn'] > 0])/len(df):.1f}%)")
    print(f"Cases with all zeros: {len(df[(df['tp'] == 0) & (df['fp'] == 0) & (df['fn'] == 0)])} ({100*len(df[(df['tp'] == 0) & (df['fp'] == 0) & (df['fn'] == 0)])/len(df):.1f}%)")
    
    print("\n" + "="*80)
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_confusion_matrix.py <path_to_summary.csv> [output_dir]")
        print("\nExample:")
        print("  python visualize_confusion_matrix.py results/intra_shift_091025/PPO_chase_2x2_20251008_015945/targeted_shift_test_summary.csv")
        sys.exit(1)
    
    summary_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "confusion_matrix_analysis"
    
    if not os.path.exists(summary_csv):
        print(f"ERROR: File not found: {summary_csv}")
        sys.exit(1)
    
    visualize_confusion_distribution(summary_csv, output_dir)
