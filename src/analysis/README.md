# ATC Hallucination Analysis Visualization Package

This comprehensive visualization package provides end-to-end analysis capabilities for aircraft collision avoidance systems and hallucination detection research.

## Overview

The package includes:

- **Geographic visualization** with folium-based interactive maps
- **Temporal analysis** with plotly interactive plots
- **Publication-quality figures** using matplotlib  
- **Hotspot clustering** and similarity analysis
- **Safety margin statistics** and performance metrics
- **Data ingestion pipeline** for harmonizing results
- **Complete orchestration** for thesis figure generation

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `folium>=0.14.0` - Interactive geographic maps
- `plotly>=5.15.0` - Interactive temporal plots
- `matplotlib>=3.6.0` - Publication figures
- `scikit-learn>=1.2.0` - Clustering analysis
- `pandas>=1.5.0` - Data processing

## Quick Start

### Generate All Figures (Recommended)

Run the complete analysis pipeline:

```bash
python analysis/make_all_figures.py --results-dir results
```

This creates:
- Geographic maps with trajectory overlays and safety analysis
- Interactive temporal analysis plots
- Publication-quality static figures  
- Comprehensive analysis reports

### Individual Components

#### Geographic Maps
```python
from analysis.viz_geographic import build_map
import pandas as pd

# Load trajectory data
baseline_df = pd.read_csv("results/trajectory_baseline.csv")
shifted_df = pd.read_csv("results/trajectory_shifted.csv")

# Generate interactive map
build_map(baseline_df, shifted_df, "comparison_map.html")
```

#### Interactive Plots
```python
from analysis.viz_plotly import time_series_panel, animated_geo

# Time series analysis
fig = time_series_panel(shifted_df, "KPI Analysis")
fig.write_html("time_series.html")

# Animated trajectories
anim_fig = animated_geo(shifted_df, "Aircraft Animation")
anim_fig.write_html("animation.html")
```

#### Publication Figures
```python
from analysis.viz_matplotlib import create_publication_figure_set

# Generate complete figure set
figures = create_publication_figure_set(summary_df, trajectory_df, "figures/")
```

#### Data Loading
```python
from analysis.ingest import load_complete_dataset

# Load and harmonize all data
dataset = load_complete_dataset("results/")
trajectories = dataset['trajectories']
summary = dataset['summary']
```

## Data Format Requirements

### Trajectory Data
Expected columns:
- `agent_id`: Aircraft identifier (A0, A1, A2, A3)
- `step_idx`: Simulation step number
- `sim_time_s`: Simulation time in seconds
- `lat_deg`, `lon_deg`: Position in degrees
- `hdg_deg`: Heading in degrees
- `tas_kt`: True airspeed in knots
- `min_separation_nm`: Minimum separation distance
- `conflict_flag`: 1 if conflict detected, 0 otherwise
- `predicted_alert`: 1 if alert predicted, 0 otherwise

### Hallucination Data
Expected columns:
- `fp`, `fn`, `tp`, `tn`: False positive, false negative, true positive, true negative flags
- `episode_id`: Episode identifier
- Matching temporal/spatial keys for merging

### Summary Data
Expected columns:
- `shift_magnitude`: Magnitude of input shift
- `shift_type`: Type of shift (heading, speed, position)
- `scenario`: Scenario name (parallel, head_on, converging)
- Performance metrics: `fn_rate`, `fp_rate`, `resolution_failure_rate`

## Output Structure

```
figures_output_YYYYMMDD_HHMMSS/
├── geographic_maps/           # Interactive folium maps
│   ├── overview_trajectories.html
│   ├── scenario_*.html
│   ├── baseline_vs_shifted.html
│   └── conflict_hotspots.html
├── interactive_plots/         # Plotly interactive figures
│   ├── time_series_kpi.html
│   ├── animated_trajectories.html
│   ├── hallucination_dashboard.html
│   └── comprehensive_report.html
├── publication_figures/       # High-quality static figures
│   ├── degradation_fn_rate.png
│   ├── agent_vulnerability.png
│   ├── confusion_evolution.png
│   ├── oscillation_patterns.png
│   └── safety_margins.png
├── analysis_results/          # Analysis outputs
│   ├── hotspot_analysis.json
│   ├── conflict_zones.json
│   ├── safety_metrics.csv
│   └── trajectory_similarity.json
├── processed_data/            # Harmonized datasets
│   ├── trajectories_harmonized.csv
│   ├── summary_harmonized.csv
│   └── metadata.json
└── reports/                   # Summary reports
    └── analysis_summary.html
```

## Key Features

### Geographic Visualizations
- **Trajectory overlays**: Baseline vs shifted trajectories with agent-specific colors
- **Safety circles**: 5 NM safety zones around aircraft positions
- **Conflict heatmaps**: Heat maps of Loss of Separation events
- **Hallucination markers**: FP/FN/TP markers with color coding
- **Time animations**: TimestampedGeoJson with time slider control
- **Hotspot analysis**: Clustering of problematic locations

### Interactive Analysis
- **Time series panels**: Min separation, DCPA, alert states over time
- **Animated maps**: Aircraft movement with interactive controls
- **Performance dashboards**: Hallucination metrics and confusion matrices
- **Degradation curves**: Performance vs shift magnitude with confidence intervals

### Publication Figures
- **Bootstrap confidence intervals**: Statistically robust uncertainty quantification
- **Agent vulnerability matrices**: Heatmaps showing agent-specific performance
- **Confusion matrix evolution**: How detection accuracy changes with shifts
- **Safety margin distributions**: Statistical analysis of separation distances
- **Action oscillation patterns**: Behavioral stability analysis

### Analysis Components
- **Hotspot clustering**: DBSCAN clustering of FP/FN events
- **Trajectory similarity**: DTW-based trajectory comparison
- **Safety statistics**: Comprehensive safety margin analysis
- **Performance degradation**: Baseline vs shifted comparison

## Advanced Usage

### Custom Analysis Pipelines
```python
from analysis.analysis_hotspots import analyze_hotspots
from analysis.analysis_similarity import analyze_trajectory_clustering
from analysis.analysis_safety import calculate_safety_metrics

# Hotspot analysis
hotspots = analyze_hotspots(df, ['fp', 'fn'], eps_nm=3.0, min_samples=5)

# Trajectory clustering
similarity = analyze_trajectory_clustering(baseline_df, shifted_df)

# Safety metrics
safety = calculate_safety_metrics(df)
```

### Command Line Options
```bash
# Skip specific components
python analysis/make_all_figures.py --skip-maps --skip-interactive

# Custom directories
python analysis/make_all_figures.py --results-dir custom_results --output-dir custom_output

# Help
python analysis/make_all_figures.py --help
```

## Research Applications

This package is designed for:

- **Thesis figure generation**: Complete set of publication-ready visualizations
- **Hallucination analysis**: Systematic study of false positive/negative patterns
- **Safety assessment**: Quantitative evaluation of collision avoidance performance
- **Behavioral analysis**: Understanding how input perturbations affect aircraft behavior
- **Geographic analysis**: Spatial patterns in conflicts and detection failures

## Performance Notes

- **Geographic maps**: Use sparse sampling for safety circles (every_n parameter) to improve performance
- **Time animations**: Limited to reasonable time ranges to avoid browser memory issues
- **Large datasets**: Consider filtering or sampling for interactive visualizations
- **Static figures**: Full resolution suitable for publication (300 DPI)

## Troubleshooting

**Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

**Empty visualizations**: Check data format and column names match expectations

**Performance issues**: Reduce data size or increase sampling intervals for interactive components

**Memory errors**: Use smaller time windows or geographic regions for animations

## Contributing

The package is modular and extensible:

- Add new visualization types in respective modules
- Extend analysis components with domain-specific metrics
- Customize styling and color schemes in individual functions
- Add new data ingestion formats in `ingest.py`

## License

This visualization package is part of the ATC Hallucination research project.