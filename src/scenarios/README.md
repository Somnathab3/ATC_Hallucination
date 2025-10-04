# Air Traffic Scenario Generation

This module generates **standardized aircraft conflict scenarios** for MARL training and robustness evaluation. All scenarios create inevitable conflicts without intervention, designed to test collision avoidance capabilities systematically.

---

## 📋 Overview

### Key Features

- **5 Standardized Geometries**: Head-on, T-formation, Parallel, Converging, Canonical Crossing
- **Reproducible Conflicts**: Fixed distances ensure convergence within episode limits
- **Parametric Generation**: Customizable spacing, speeds, and aircraft types
- **Interactive Visualization**: HTML radar plots with Plotly for presentation
- **JSON Export**: Scenario configurations compatible with environment loader

### Design Philosophy

All scenarios follow these principles:

1. **Deterministic**: Same seed produces identical initial conditions
2. **Centered**: All scenarios use common airspace center (52.0°N, 4.0°E)
3. **Balanced**: Symmetric conflicts prevent single-agent bias
4. **Convergent**: "FIXED" distances ensure episodes complete within 1000 seconds
5. **Scalable**: Parameters adjust conflict severity for difficulty variation

---

## 🎯 Scenario Specifications

### 1. Head-On Encounter

**Geometry:** Two aircraft approaching on reciprocal headings

```python
from src.scenarios.scenario_generator import make_head_on
scenario_path = make_head_on(approach_nm=18.0)
```

**Parameters:**
- `approach_nm`: Initial separation distance from center (default: 18.0)

**Layout:**
```
    A2 ↓ (hdg=180°)
        |
    [CENTER]
        |
    A1 ↑ (hdg=0°)
```

**Conflict Characteristics:**
- **Closure rate**: 2× cruise speed (~500 kt combined)
- **TCPA**: ~2-3 minutes without intervention
- **Difficulty**: Medium (2 agents, simple geometry)

**Training Focus:**
- Basic conflict detection and resolution
- Heading change timing optimization
- Reciprocal maneuver coordination

---

### 2. T-Formation Crossing

**Geometry:** Horizontal bar + vertical stem perpendicular intersection

```python
from src.scenarios.scenario_generator import make_t_formation
scenario_path = make_t_formation(arm_nm=7.5, stem_nm=10.0)
```

**Parameters:**
- `arm_nm`: Distance from center to each arm aircraft (default: 7.5)
- `stem_nm`: Distance from center to stem aircraft (default: 10.0)

**Layout:**
```
    A3 ↓ (hdg=180°, stem)
        |
A1 → [CENTER] ← A2
(hdg=90°)  (hdg=270°)
```

**Conflict Characteristics:**
- **Three-way conflict**: Stem aircraft conflicts with both arm aircraft
- **Sequential resolution**: Arm aircraft can coordinate laterally
- **TCPA**: ~1.5-2 minutes for stem aircraft
- **Difficulty**: Hard (3 agents, asymmetric roles)

**Training Focus:**
- Multi-agent coordination
- Prioritization under asymmetric threat
- Sequenced maneuver timing

---

### 3. Parallel In-Trail

**Geometry:** Same-direction traffic with in-trail spacing challenges

```python
from src.scenarios.scenario_generator import make_parallel
scenario_path = make_parallel(gaps_nm=8.0, south_nm=18.0)
```

**Parameters:**
- `gaps_nm`: In-trail spacing between aircraft (default: 8.0)
- `south_nm`: Distance south of center for southernmost aircraft (default: 18.0)

**Layout:**
```
All heading north (hdg=0°):
    A3 ↑  (northernmost)
    
    A2 ↑  (middle, +8 NM)
    
    A1 ↑  (southernmost, +8 NM)
```

**Conflict Characteristics:**
- **Longitudinal separation**: In-trail spacing maintenance
- **Speed-based resolution**: Leading aircraft must maintain pace
- **No lateral conflict**: All on same track
- **Difficulty**: Medium (3 agents, speed coordination)

**Training Focus:**
- Speed control mastery
- In-trail separation maintenance
- Energy management (acceleration/deceleration)

---

### 4. Converging Encounter

**Geometry:** Multiple aircraft targeting nearby waypoints

```python
from src.scenarios.scenario_generator import make_converging
scenario_path = make_converging(radius_nm=12.0, wp_cluster_nm=2.0)
```

**Parameters:**
- `radius_nm`: Initial placement radius around center (default: 12.0)
- `wp_cluster_nm`: Waypoint clustering radius (default: 2.0)

**Layout:**
```
    A1 (NE) ↘
               ↘
A4 (W) →    [CENTER]    ← A2 (E)
               ↗
    A3 (SW) ↗
```

**Conflict Characteristics:**
- **Four-way convergence**: All aircraft heading toward center cluster
- **Sequential conflicts**: Pairwise conflicts emerge at different times
- **High coordination requirement**: 4 agents must deconflict simultaneously
- **Difficulty**: Very Hard (4 agents, complex interactions)

**Training Focus:**
- Multi-agent conflict resolution
- Sequential deconfliction
- Waypoint precision under constraint

---

### 5. Canonical Crossing

**Geometry:** Orthogonal four-way intersection (classic ATC scenario)

```python
from src.scenarios.scenario_generator import make_canonical_crossing
scenario_path = make_canonical_crossing(radius_nm=12.5)
```

**Parameters:**
- `radius_nm`: Distance from center to each aircraft (default: 12.5)

**Layout:**
```
    A1 ↓ (N, hdg=180°)
        |
A4 → [CENTER] ← A2
(W)     |      (E)
    A3 ↑ (S, hdg=0°)
```

**Conflict Characteristics:**
- **Symmetric conflicts**: All agents face equivalent threat profiles
- **Simultaneous arrival**: All TCPA values similar
- **No priority**: Requires cooperative resolution
- **Difficulty**: Very Hard (4 agents, perfect symmetry)

**Training Focus:**
- Symmetric cooperative policy
- Priority-free deconfliction
- Emergent coordination patterns

---

## 📐 Geometric Calculations

### Distance Conversion Functions

**Nautical miles to latitude degrees:**
$$\Delta \text{lat} = \frac{\text{NM}}{60}$$

**Nautical miles to longitude degrees (latitude-corrected):**
$$\Delta \text{lon} = \frac{\text{NM}}{60 \cdot \cos(\text{lat}_{ref})}$$

**Implementation:**
```python
def nm_to_lat(nm: float) -> float:
    return nm / 60.0

def nm_to_lon(nm: float, at_lat_deg: float) -> float:
    return nm / (60.0 * max(1e-6, math.cos(math.radians(at_lat_deg))))
```

### Position Offset Calculation

Given base position `(lat0, lon0)` and offsets in NM:

```python
def pos_offset(lat0, lon0, north_nm, east_nm):
    """Return new position with north/east offsets in nautical miles."""
    new_lat = lat0 + nm_to_lat(north_nm)
    new_lon = lon0 + nm_to_lon(east_nm, lat0)
    return (new_lat, new_lon)
```

**Example:**
```python
center = (52.0, 4.0)  # Netherlands airspace
pos_10nm_north = pos_offset(52.0, 4.0, north_nm=10, east_nm=0)
# Result: (52.1667, 4.0)
```

### Waypoint Placement

**Mirror position across center:**
```python
wp_lat = center_lat + (center_lat - start_lat)
wp_lon = center_lon + (center_lon - start_lon)
```

This ensures aircraft must traverse the conflict zone to reach their destination.

---

## 📄 JSON Scenario Format

Generated scenarios are saved as JSON files:

```json
{
  "scenario_name": "head_on",
  "seed": 42,
  "notes": "Two-aircraft head-on encounter...",
  "center": {
    "lat": 52.0,
    "lon": 4.0,
    "alt_ft": 10000.0
  },
  "sim_dt_s": 1.0,
  "agents": [
    {
      "id": "A1",
      "type": "A320",
      "lat": 51.7,
      "lon": 4.0,
      "hdg_deg": 0.0,
      "spd_kt": 250.0,
      "alt_ft": 10000.0,
      "waypoint": {
        "lat": 52.3,
        "lon": 4.0
      }
    },
    {
      "id": "A2",
      "type": "A320",
      "lat": 52.3,
      "lon": 4.0,
      "hdg_deg": 180.0,
      "spd_kt": 250.0,
      "alt_ft": 10000.0,
      "waypoint": {
        "lat": 51.7,
        "lon": 4.0
      }
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `scenario_name` | string | Unique scenario identifier |
| `seed` | integer | Random seed (for reproducibility) |
| `notes` | string | Human-readable description |
| `center.lat` | float | Airspace center latitude (degrees) |
| `center.lon` | float | Airspace center longitude (degrees) |
| `center.alt_ft` | float | Flight level altitude (feet) |
| `sim_dt_s` | float | BlueSky simulation timestep (seconds) |
| `agents` | array | List of aircraft configurations |
| `agents[].id` | string | Unique aircraft identifier (A1, A2, ...) |
| `agents[].type` | string | Aircraft type (A320, B738, etc.) |
| `agents[].lat` | float | Initial latitude (degrees) |
| `agents[].lon` | float | Initial longitude (degrees) |
| `agents[].hdg_deg` | float | Initial heading (degrees, 0-360) |
| `agents[].spd_kt` | float | Initial speed (knots) |
| `agents[].alt_ft` | float | Initial altitude (feet) |
| `agents[].waypoint.lat` | float | Destination latitude |
| `agents[].waypoint.lon` | float | Destination longitude |

---

## 📊 Interactive Visualization

### Radar Plots (Plotly HTML)

The module includes presentation-grade interactive visualizations:

```python
from src.scenarios.plotly_scenario_visualizer_presentation_grade import visualize_all_scenarios

# Generate all radar plots
visualize_all_scenarios(
    scenarios_dir="scenarios",
    output_dir="scenarios/scenario_plots"
)
```

**Output:** `scenario_plots/{scenario_name}_radar.html`

**Features:**
- Aircraft symbols with heading arrows
- Dashed lines showing initial heading vectors
- Solid lines showing paths to waypoints
- Conflict zone highlighting (5 NM radius)
- Interactive hover tooltips (aircraft ID, position, speed)
- Zoom/pan controls

**Example usage in presentations:**
- Open HTML file in browser
- Take screenshot or embed in slides
- Interactive demo during talks

### Static Plots (Matplotlib)

For publication figures:

```python
from src.scenarios.visualize_scenarios import visualize_scenario_static

visualize_scenario_static(
    scenario_path="scenarios/head_on.json",
    output_path="figures/head_on_static.png",
    dpi=300
)
```

---

## 🚀 Usage Examples

### Generate All Scenarios

```python
from src.scenarios.scenario_generator import (
    make_head_on,
    make_t_formation,
    make_parallel,
    make_converging,
    make_canonical_crossing
)

# Generate with default parameters
scenarios = [
    make_head_on(),
    make_t_formation(),
    make_parallel(),
    make_converging(),
    make_canonical_crossing()
]

print(f"Generated {len(scenarios)} scenarios:")
for path in scenarios:
    print(f"  - {path}")
```

**Output:**
```
Generated 5 scenarios:
  - scenarios/head_on.json
  - scenarios/t_formation.json
  - scenarios/parallel.json
  - scenarios/converging.json
  - scenarios/canonical_crossing.json
```

### Custom Parameters

```python
# Closer head-on (higher difficulty)
make_head_on(approach_nm=12.0)  # 33% shorter approach

# Wider T-formation (lower difficulty)
make_t_formation(arm_nm=10.0, stem_nm=15.0)

# Tighter parallel spacing (coordination challenge)
make_parallel(gaps_nm=5.0, south_nm=15.0)

# Larger converging radius (more time to react)
make_converging(radius_nm=18.0, wp_cluster_nm=3.0)

# Canonical crossing with extended approach
make_canonical_crossing(radius_nm=15.0)
```

### CLI Usage

```bash
# Generate all scenarios
python atc_cli.py generate-scenarios --all

# Generate single scenario with custom parameters
python atc_cli.py generate-scenarios --scenario head_on --params "approach_nm=15.0"

# Visualize generated scenarios
python src/scenarios/plotly_scenario_visualizer_presentation_grade.py
```

### Loading Scenarios in Environment

```python
from src.environment.marl_collision_env_minimal import MARLCollisionEnv

env = MARLCollisionEnv({
    "scenario_path": "scenarios/head_on.json",
    "seed": 42
})

obs, info = env.reset()
# Environment now initialized with head-on conflict geometry
```

---

## 🔧 Advanced Customization

### Creating Custom Scenarios

```python
import json
import os

# Define custom scenario
custom_scenario = {
    "scenario_name": "my_custom_scenario",
    "seed": 123,
    "notes": "Three-aircraft oblique crossing",
    "center": {"lat": 52.0, "lon": 4.0, "alt_ft": 10000.0},
    "sim_dt_s": 1.0,
    "agents": [
        {
            "id": "A1",
            "type": "A320",
            "lat": 51.8,
            "lon": 3.8,
            "hdg_deg": 45.0,  # Northeast
            "spd_kt": 250.0,
            "alt_ft": 10000.0,
            "waypoint": {"lat": 52.2, "lon": 4.2}
        },
        # Add more agents...
    ]
}

# Save to file
os.makedirs("scenarios", exist_ok=True)
with open("scenarios/my_custom_scenario.json", "w") as f:
    json.dump(custom_scenario, f, indent=2)
```

### Parametric Scenario Generation

```python
def make_diamond_crossing(radius_nm=10.0, angle_deg=45.0):
    """
    Create diamond-shaped crossing pattern.
    
    Args:
        radius_nm: Distance from center to each aircraft
        angle_deg: Angular offset from cardinal directions
    """
    from math import cos, sin, radians
    
    agents = []
    for i, (direction, hdg_offset) in enumerate([
        ("NE", 225), ("SE", 315), ("SW", 45), ("NW", 135)
    ]):
        # Position on circle
        angle_rad = radians(angle_deg + i * 90)
        north_nm = radius_nm * sin(angle_rad)
        east_nm = radius_nm * cos(angle_rad)
        
        start_lat, start_lon = pos_offset(CENTER_LAT, CENTER_LON, north_nm, east_nm)
        
        # Waypoint is mirror position
        wp_lat = CENTER_LAT - north_nm / 60.0
        wp_lon = CENTER_LON - east_nm / (60.0 * cos(radians(CENTER_LAT)))
        
        agents.append({
            "id": f"A{i+1}",
            "type": "A320",
            "lat": start_lat,
            "lon": start_lon,
            "hdg_deg": hdg_offset,
            "spd_kt": 250.0,
            "alt_ft": 10000.0,
            "waypoint": {"lat": wp_lat, "lon": wp_lon}
        })
    
    return save("diamond_crossing", agents, "Four-aircraft diamond crossing pattern")
```

### Variable Speed Scenarios

```python
def make_speed_differential_head_on(approach_nm=18.0, speed_diff_kt=50.0):
    """
    Head-on with asymmetric speeds for testing speed-based resolution.
    """
    a1_lat, a1_lon = pos_offset(CENTER_LAT, CENTER_LON, -approach_nm, 0.0)
    a2_lat, a2_lon = pos_offset(CENTER_LAT, CENTER_LON, +approach_nm, 0.0)
    
    agents = [
        {
            "id": "A1",
            "type": "A320",
            "lat": a1_lat,
            "lon": a1_lon,
            "hdg_deg": 0.0,
            "spd_kt": 250.0 - speed_diff_kt / 2,  # Slower
            "alt_ft": 10000.0,
            "waypoint": {"lat": a2_lat, "lon": a2_lon}
        },
        {
            "id": "A2",
            "type": "B738",
            "lat": a2_lat,
            "lon": a2_lon,
            "hdg_deg": 180.0,
            "spd_kt": 250.0 + speed_diff_kt / 2,  # Faster
            "alt_ft": 10000.0,
            "waypoint": {"lat": a1_lat, "lon": a1_lon}
        }
    ]
    
    return save("speed_differential_head_on", agents,
                f"Head-on with {speed_diff_kt} kt speed differential")
```

---

## 📊 Scenario Comparison Table

| Scenario | Agents | Conflicts | Symmetry | Difficulty | Training Time |
|----------|--------|-----------|----------|------------|---------------|
| Head-On | 2 | 1 pairwise | High | ⭐⭐ Medium | ~20 min |
| T-Formation | 3 | 2 pairwise | Low | ⭐⭐⭐ Hard | ~35 min |
| Parallel | 3 | 2 in-trail | Medium | ⭐⭐ Medium | ~30 min |
| Converging | 4 | 6 pairwise | Medium | ⭐⭐⭐⭐ V.Hard | ~50 min |
| Canonical | 4 | 4 pairwise | Perfect | ⭐⭐⭐⭐ V.Hard | ~55 min |

*Training time estimates for 100k timesteps on RTX 3080*

### Recommended Training Sequence

1. **Head-On** (foundational): Learn basic conflict detection and heading changes
2. **Parallel** (speed control): Master longitudinal separation via speed
3. **T-Formation** (coordination): Handle asymmetric multi-agent conflicts
4. **Canonical Crossing** (symmetry): Develop cooperative priority-free resolution
5. **Converging** (complexity): Tackle sequential multi-agent deconfliction

---

## 🐛 Troubleshooting

### Common Issues

**1. Scenarios generate outside episode time limit**
```python
# Problem: Aircraft can't reach waypoints in 1000 seconds
# Solution: Reduce distances or increase speeds

make_head_on(approach_nm=12.0)  # Shorter approach
# Or modify scenario JSON: increase spd_kt to 300
```

**2. Conflicts resolve too easily**
```python
# Problem: Agents avoid conflicts without learning
# Solution: Tighter initial positions

make_converging(radius_nm=8.0)  # Closer convergence
make_canonical_crossing(radius_nm=10.0)  # Tighter crossing
```

**3. Waypoint placement errors**
```python
# Problem: Waypoints not aligned with scenario geometry
# Solution: Use mirror position calculation

wp_lat = center_lat + (center_lat - start_lat)
wp_lon = center_lon + (center_lon - start_lon)
```

**4. Visualization not showing**
```bash
# Problem: Missing Plotly or Folium dependencies
# Solution: Install visualization packages

pip install plotly>=5.15.0 folium>=0.14.0 kaleido>=0.2.1
```

### Validation Checks

```python
def validate_scenario(scenario_path):
    """Validate scenario JSON for common issues."""
    with open(scenario_path) as f:
        scenario = json.load(f)
    
    # Check required fields
    required = ['scenario_name', 'agents', 'center']
    for field in required:
        assert field in scenario, f"Missing field: {field}"
    
    # Check agent count
    assert len(scenario['agents']) >= 2, "Need at least 2 agents"
    
    # Check unique IDs
    ids = [a['id'] for a in scenario['agents']]
    assert len(ids) == len(set(ids)), "Duplicate agent IDs"
    
    # Check waypoints exist
    for agent in scenario['agents']:
        assert 'waypoint' in agent, f"Agent {agent['id']} missing waypoint"
    
    print(f"✓ Scenario {scenario['scenario_name']} validated successfully")
```

---

## 📚 References

- **Scenario design principles**: Inspired by FAA ATC training materials
- **Geometric calculations**: Based on aviation navigation standards (great circle distance)
- **Visualization**: Plotly Express documentation for interactive figures

---

**Module Files:**
- `scenario_generator.py`: Core generation functions (250 lines)
- `plotly_scenario_visualizer_presentation_grade.py`: Interactive HTML visualizations
- `visualize_scenarios.py`: Static Matplotlib plots

**Related Modules:**
- [src/environment/](../environment/README.md): Environment loads scenarios
- [src/testing/](../testing/README.md): Distribution shift scenario modifications
- [src/analysis/](../analysis/README.md): Trajectory visualization

---

**Generated Scenarios Location:** `scenarios/*.json`  
**Visualization Output:** `scenarios/scenario_plots/*.html`
