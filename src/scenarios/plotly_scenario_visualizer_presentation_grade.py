"""
plotly_scenario_viz_radar.py — clean, radar‑style Plotly visuals for ATC scenarios

Changes vs previous:
- Smaller, auto‑scaled aircraft triangles (size in NM tied to scene span).
- Non‑overlapping metadata box anchored to paper coords.
- Soft white theme, light grid, dashed paths, subtler fills.
- Optional radar rings (5/10/15/20 NM) around sector center.
- Better label placement and compact fonts.
- PRESENTATION GRADE: Enhanced font sizes (16px+ for all text)
- COMBINED VISUALIZATION: All scenarios in single PNG for presentations

Usage:
  # Single scenario
  python plotly_scenario_viz_radar.py --file scenarios/head_on.json --out plots
  
  # All scenarios individually  
  python plotly_scenario_viz_radar.py --scenarios-dir scenarios --out plots
  
  # Combined presentation plot (ALL scenarios in one image)
  python plotly_scenario_viz_radar.py --combined --scenarios-dir scenarios --out plots

Combined plot features:
- All scenarios in organized subplots
- Optimized for presentation slides
- High-resolution PNG output (2000x1500)
- Professional typography and spacing
- Perfect for academic papers and executive summaries

Tip: install `kaleido` for PNG export.
"""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- geo helpers -----------------
CENTER_LAT_DEFAULT = 52.0


def nm_to_lat(nm: float) -> float:
    return nm / 60.0


def nm_to_lon(nm: float, at_lat_deg: float) -> float:
    return nm / (60.0 * max(1e-6, math.cos(math.radians(at_lat_deg))))


def latlon_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dN = (lat2 - lat1) * 60.0
    dE = (lon2 - lon1) * 60.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dN, dE)


# --------------- shapes & transforms ---------------

def _rotate_en(e_nm: float, n_nm: float, hdg_deg: float) -> Tuple[float, float]:
    """Rotate an (east, north) vector by **−hdg_deg** (CW from North → correct aircraft heading).
    Inputs/outputs are in nautical miles within the local EN frame.


    Sanity:
    hdg= 0° → nose to North (0, +1)
    hdg= 90° → nose to East (+1, 0)
    hdg= 180° → nose to South (0, −1)
    hdg= 270° → nose to West (−1, 0)
    """
    alpha = -math.radians(hdg_deg) # rotate CW by heading
    c, s = math.cos(alpha), math.sin(alpha)
    e_rot = e_nm * c - n_nm * s
    n_rot = e_nm * s + n_nm * c
    return e_rot, n_rot


def aircraft_triangle_points(lat: float, lon: float, hdg_deg: float, size_nm: float) -> List[Tuple[float, float]]:
    # slim, presentation‑friendly triangle
    pts_en = [
        (0.0, size_nm * 1.3),          # nose
        (-size_nm * 0.7, -size_nm * 0.6),
        ( size_nm * 0.7, -size_nm * 0.6),
    ]
    poly: List[Tuple[float, float]] = []
    for e, n in pts_en:
        e_r, n_r = _rotate_en(e, n, hdg_deg)
        poly.append((lon + nm_to_lon(e_r, lat), lat + nm_to_lat(n_r)))
    poly.append(poly[0])  # close
    return poly


def diamond_points(lat: float, lon: float, size_nm: float) -> List[Tuple[float, float]]:
    return [
        (lon,                            lat + nm_to_lat(size_nm)),
        (lon + nm_to_lon(size_nm, lat),  lat),
        (lon,                            lat - nm_to_lat(size_nm)),
        (lon - nm_to_lon(size_nm, lat),  lat),
        (lon,                            lat + nm_to_lat(size_nm)),
    ]


# --------------- drawing helpers ---------------
COLORS = ["#2E86DE", "#10AC84", "#EE5253", "#F39C12", "#8E44AD", "#16A085"]


def _add_polygon(fig: go.Figure, pts: List[Tuple[float, float]], fill: str, name: str,
                 opacity: float = 0.85, lw: float = 1.5):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(width=lw, color="#111"),
        fill="toself", fillcolor=fill, opacity=opacity, name=name,
        showlegend=False, hoverinfo="skip"))


def _add_heading(fig: go.Figure, lat: float, lon: float, hdg: float, length_nm: float, color: str):
    e, n = _rotate_en(0.0, length_nm, hdg)
    lat2 = lat + nm_to_lat(n)
    lon2 = lon + nm_to_lon(e, lat)
    fig.add_trace(go.Scatter(x=[lon, lon2], y=[lat, lat2], mode="lines",
                             line=dict(width=2, color=color), showlegend=False, hoverinfo="skip"))


def _add_path(fig: go.Figure, lat: float, lon: float, wp_lat: float, wp_lon: float, color: str) -> float:
    d = latlon_distance_nm(lat, lon, wp_lat, wp_lon)
    fig.add_trace(go.Scatter(x=[lon, wp_lon], y=[lat, wp_lat], mode="lines",
                             line=dict(width=2, dash="dash", color=color), showlegend=False, hoverinfo="skip"))
    # midpoint label
    mlon, mlat = (lon + wp_lon) / 2.0, (lat + wp_lat) / 2.0
    fig.add_annotation(x=mlon, y=mlat, text=f"{d:.1f} NM", showarrow=False,
                       font=dict(size=14, family="Arial, sans-serif"), bgcolor="white", bordercolor=color, borderwidth=2, opacity=0.95)
    return d


def _label_offset(center_lat: float, center_lon: float, lat: float, lon: float, base_nm: float) -> Tuple[float, float, str, str]:
    e_nm = base_nm if lon >= center_lon else -base_nm
    n_nm = base_nm if lat >= center_lat else -base_nm
    return e_nm, n_nm, ("left" if e_nm > 0 else "right"), ("bottom" if n_nm > 0 else "top")


def _waypoint_label_offset(center_lat: float, center_lon: float, lat: float, lon: float, 
                          aircraft_lat: float, aircraft_lon: float, base_nm: float, 
                          all_aircraft_positions: Optional[List[Tuple[float, float]]] = None) -> Tuple[float, float, str, str]:
    """Calculate waypoint label offset that avoids aircraft label collision"""
    # Start with standard offset
    e_nm = base_nm if lon >= center_lon else -base_nm
    n_nm = base_nm if lat >= center_lat else -base_nm
    
    # Check if this would overlap with aircraft label position
    aircraft_e_nm = base_nm if aircraft_lon >= center_lon else -base_nm
    aircraft_n_nm = base_nm if aircraft_lat >= center_lat else -base_nm
    
    # Check collision with own aircraft
    if (e_nm > 0) == (aircraft_e_nm > 0) and (n_nm > 0) == (aircraft_n_nm > 0):
        distance_to_aircraft = latlon_distance_nm(lat, lon, aircraft_lat, aircraft_lon)
        if distance_to_aircraft < base_nm * 2:  # Close proximity
            # Put waypoint label on opposite side
            e_nm = -e_nm
            n_nm = -n_nm
    
    # Check collision with other aircraft if positions provided
    if all_aircraft_positions:
        for other_lat, other_lon in all_aircraft_positions:
            if other_lat == aircraft_lat and other_lon == aircraft_lon:
                continue  # Skip own aircraft
            
            # Calculate where other aircraft's label would be
            other_e_nm = base_nm if other_lon >= center_lon else -base_nm
            other_n_nm = base_nm if other_lat >= center_lat else -base_nm
            
            # Check if waypoint is very close to other aircraft
            distance_to_other = latlon_distance_nm(lat, lon, other_lat, other_lon)
            if distance_to_other < base_nm * 1.5:  # Very close to another aircraft
                # Check if waypoint label would be in same quadrant as other aircraft's label
                if (e_nm > 0) == (other_e_nm > 0) and (n_nm > 0) == (other_n_nm > 0):
                    # Try perpendicular positioning
                    if abs(lat - other_lat) > abs(lon - other_lon):  # More vertical separation
                        e_nm = 0  # Center horizontally
                        n_nm = base_nm * 1.8 if lat > center_lat else -base_nm * 1.8  # Increase vertical offset
                    else:  # More horizontal separation  
                        n_nm = 0  # Center vertically
                        e_nm = base_nm * 1.8 if lon > center_lon else -base_nm * 1.8  # Increase horizontal offset
    
    return e_nm, n_nm, ("left" if e_nm > 0 else "right"), ("bottom" if n_nm > 0 else "top")


def add_aircraft(fig: go.Figure, ag: Dict[str, Any], color: str, center_lat: float, center_lon: float,
                 sym_size_nm: float, info_gap_nm: float):
    lat, lon, hdg = float(ag["lat"]), float(ag["lon"]), float(ag["hdg_deg"])
    _add_polygon(fig, aircraft_triangle_points(lat, lon, hdg, sym_size_nm), color, f"AC {ag['id']}")
    _add_heading(fig, lat, lon, hdg, length_nm=sym_size_nm * 1.0, color=color)

    e_nm, n_nm, xa, ya = _label_offset(center_lat, center_lon, lat, lon, info_gap_nm)
    fig.add_annotation(
        x=lon + nm_to_lon(e_nm, lat), y=lat + nm_to_lat(n_nm),
        text=(f"<b>{ag['id']}</b><br>HDG: {hdg:.0f}°<br>SPD: {ag['spd_kt']:.0f} kt"
              f"<br>ALT: {ag['alt_ft']:.0f} ft"),
        showarrow=False, xanchor=xa, yanchor=ya,
        bgcolor="white", bordercolor="#222", borderwidth=2, opacity=0.95,
        font=dict(size=16, family="Arial, sans-serif"))


def add_waypoint(fig: go.Figure, ag: Dict[str, Any], color: str, center_lat: float, center_lon: float,
                 size_nm: float, info_gap_nm: float, all_aircraft_positions: Optional[List[Tuple[float, float]]] = None):
    if "waypoint" not in ag:
        return
    wp_lat = float(ag["waypoint"]["lat"]) ; wp_lon = float(ag["waypoint"]["lon"])
    aircraft_lat = float(ag["lat"]) ; aircraft_lon = float(ag["lon"])
    
    _add_polygon(fig, diamond_points(wp_lat, wp_lon, size_nm), color, f"WP-{ag['id']}", opacity=0.65, lw=1.2)
    fig.add_trace(go.Scatter(x=[wp_lon], y=[wp_lat], mode="markers",
                             marker=dict(size=7, color="white", line=dict(width=1, color="#111")),
                             showlegend=False, hoverinfo="skip"))
    
    # Use specialized waypoint offset to avoid aircraft label collision
    e_nm, n_nm, xa, ya = _waypoint_label_offset(center_lat, center_lon, wp_lat, wp_lon, 
                                               aircraft_lat, aircraft_lon, info_gap_nm * 0.8,
                                               all_aircraft_positions)
    fig.add_annotation(x=wp_lon + nm_to_lon(e_nm, wp_lat), y=wp_lat + nm_to_lat(n_nm),
                       text=f"<b>WP-{ag['id']}</b><br>LAT: {wp_lat:.3f}°<br>LON: {wp_lon:.3f}°",
                       showarrow=False, xanchor=xa, yanchor=ya,
                       bgcolor=color, bordercolor="#222", borderwidth=2, opacity=0.9,
                       font=dict(size=15, color="white", family="Arial, sans-serif"))


def add_radar_rings(fig: go.Figure, center_lat: float, center_lon: float, radii_nm=(5, 10, 15, 20)):
    for r in radii_nm:
        xs, ys = [], []
        for k in range(0, 361, 3):
            th = math.radians(k)
            e_nm, n_nm = r * math.cos(th), r * math.sin(th)
            xs.append(center_lon + nm_to_lon(e_nm, center_lat))
            ys.append(center_lat + nm_to_lat(n_nm))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color="#C9D3E0", width=1), showlegend=False, hoverinfo="skip"))
        # ring label
        fig.add_annotation(x=center_lon + nm_to_lon(r, center_lat), y=center_lat,
                           text=f"{r} NM", showarrow=False, font=dict(size=13, color="#5A6C85", family="Arial, sans-serif"))


# --------------- main fig ---------------

def _bounds(scen: Dict[str, Any]) -> Tuple[float, float, float, float]:
    lats, lons = [], []
    center = scen.get("center", {"lat": CENTER_LAT_DEFAULT, "lon": 4.0})
    lats.append(float(center["lat"])) ; lons.append(float(center["lon"]))
    for ag in scen.get("agents", []):
        lats.append(float(ag["lat"])) ; lons.append(float(ag["lon"]))
        if "waypoint" in ag:
            lats.append(float(ag["waypoint"]["lat"]))
            lons.append(float(ag["waypoint"]["lon"]))
    return min(lats), max(lats), min(lons), max(lons)


def _scene_span_nm(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    mean_lat = (lat_min + lat_max) / 2.0
    dN = (lat_max - lat_min) * 60.0
    dE = (lon_max - lon_min) * 60.0 * math.cos(math.radians(mean_lat))
    return max(dN, dE)


def _symbol_sizes(scen: Dict[str, Any]) -> Tuple[float, float]:
    lat_min, lat_max, lon_min, lon_max = _bounds(scen)
    span_nm = max(12.0, _scene_span_nm(lat_min, lat_max, lon_min, lon_max))  # guard
    # triangle size: ~2.0 NM for compact scenes, up to 3.0 NM for larger
    tri = max(1.4, min(3.0, span_nm * 0.035))
    # label offsets
    gap = max(2.2, min(4.0, span_nm * 0.06))
    return tri, gap


def make_figure(scen: Dict[str, Any], add_rings: bool = True) -> go.Figure:
    center = scen.get("center", {"lat": CENTER_LAT_DEFAULT, "lon": 4.0})
    center_lat, center_lon = float(center["lat"]), float(center["lon"])

    tri_nm, gap_nm = _symbol_sizes(scen)

    # Collect all aircraft positions for collision detection
    all_aircraft_positions = [(float(ag["lat"]), float(ag["lon"])) for ag in scen.get("agents", [])]

    fig = go.Figure()

    # paths first
    distances = []
    for i, ag in enumerate(scen.get("agents", [])):
        color = COLORS[i % len(COLORS)]
        if "waypoint" in ag:
            d = _add_path(fig, float(ag["lat"]), float(ag["lon"]), float(ag["waypoint"]["lat"]), float(ag["waypoint"]["lon"]), color)
            distances.append(d)

    # waypoints + aircraft
    for i, ag in enumerate(scen.get("agents", [])):
        color = COLORS[i % len(COLORS)]
        add_waypoint(fig, ag, color, center_lat, center_lon, size_nm=max(0.9, tri_nm * 0.45), info_gap_nm=gap_nm, all_aircraft_positions=all_aircraft_positions)
        add_aircraft(fig, ag, color, center_lat, center_lon, sym_size_nm=tri_nm, info_gap_nm=gap_nm)

    # center marker & rings
    fig.add_trace(go.Scatter(x=[center_lon], y=[center_lat], mode="markers",
                             marker=dict(size=8, color="#111"), showlegend=False, hoverinfo="skip"))
    if add_rings:
        add_radar_rings(fig, center_lat, center_lon)

    # layout
    title_name = scen.get("scenario_name", "Scenario").replace("_", " ").title()
    notes = scen.get("notes", "")
    if len(notes) > 160:
        notes = notes[:160] + "…"

    fig.update_layout(
        title=dict(
            text=f"<b style='font-size:28px'>{title_name}</b><br><span style='font-size:18px'>{notes}</span>", 
            x=0.5,
            font=dict(family="Arial, sans-serif")
        ),
        template="plotly_white",
        margin=dict(l=90, r=90, t=120, b=90),
        width=1400, height=1000,
        showlegend=False,
    )

    # axes: equal scale, soft grid
    fig.update_xaxes(
        title_text="Longitude (deg)", 
        title_font=dict(size=18, family="Arial, sans-serif"),
        tickfont=dict(size=14),
        gridcolor="#ECF0F4", 
        zeroline=False
    )
    fig.update_yaxes(
        title_text="Latitude (deg)", 
        title_font=dict(size=18, family="Arial, sans-serif"),
        tickfont=dict(size=14),
        gridcolor="#ECF0F4", 
        zeroline=False, 
        scaleanchor="x", 
        scaleratio=1
    )

    # scenario metadata box (fixed in paper coords → won’t collide with axes)
    if distances:
        meta = (f"<b>Aircraft:</b> {len(scen.get('agents', []))}<br>"
                f"<b>Avg Dist:</b> {sum(distances)/len(distances):.1f} NM<br>"
                f"<b>Range:</b> {min(distances):.1f}-{max(distances):.1f} NM<br>"
                f"<b>Seed:</b> {scen.get('seed','N/A')}")
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                           text=meta, showarrow=False, bgcolor="#E8F1FF", bordercolor="#9BB7E1",
                           borderwidth=2, opacity=0.98, font=dict(size=14, family="Arial, sans-serif"))

    # viewport padding
    lat_min, lat_max, lon_min, lon_max = _bounds(scen)
    pad_lat = (lat_max - lat_min) * 0.25 + nm_to_lat(tri_nm * 2.0)
    pad_lon = (lon_max - lon_min) * 0.25 + nm_to_lon(tri_nm * 2.0, (lat_min + lat_max) / 2.0)
    fig.update_xaxes(range=[lon_min - pad_lon, lon_max + pad_lon])
    fig.update_yaxes(range=[lat_min - pad_lat, lat_max + pad_lat])

    return fig


# --------------- IO ---------------

def load_scenario(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(fig: go.Figure, out_dir: str | Path, name: str, save_png: bool = True) -> Tuple[str, Optional[str]]:
    out = Path(out_dir) ; out.mkdir(parents=True, exist_ok=True)
    html = str(out / f"{name}.html")
    fig.write_html(html, include_plotlyjs="cdn")
    png = None
    if save_png:
        try:
            png = str(out / f"{name}.png")
            fig.write_image(png, width=1800, height=1300, scale=2.5)
        except Exception:
            png = None
    return html, png


def plot_file(file_path: str | Path, out_dir: str | Path = "plots", png: bool = True) -> Tuple[str, Optional[str]]:
    scen = load_scenario(file_path)
    fig = make_figure(scen)
    return save_plot(fig, out_dir, Path(file_path).stem + "_radar", save_png=png)


def plot_dir(scen_dir: str | Path = "scenarios", out_dir: str | Path = "plots", png: bool = True):
    res = []
    for p in sorted(Path(scen_dir).glob("*.json")):
        res.append(plot_file(p, out_dir, png))
    return res


def create_combined_scenarios_plot(scen_dir: str | Path = "scenarios", out_dir: str | Path = "plots") -> Tuple[str, Optional[str]]:
    """
    Create a combined plot with all scenarios in separate subplots within a single figure.
    Perfect for presentation slides showing all scenarios at once.
    """
    scenario_files = sorted(Path(scen_dir).glob("*.json"))
    num_scenarios = len(scenario_files)
    
    if num_scenarios == 0:
        raise ValueError(f"No scenario files found in {scen_dir}")
    
    # Calculate subplot grid (prefer wider layout for presentations)
    if num_scenarios <= 2:
        rows, cols = 1, num_scenarios
    elif num_scenarios <= 4:
        rows, cols = 2, 2
    elif num_scenarios <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    # Create subplot titles
    scenario_titles = []
    scenarios = []
    for file_path in scenario_files:
        scen = load_scenario(file_path)
        scenarios.append(scen)
        title_name = scen.get("scenario_name", file_path.stem).replace("_", " ").title()
        scenario_titles.append(title_name)
    
    # Create subplots with individual titles
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=scenario_titles,
        specs=[[{"type": "scatter"}] * cols for _ in range(rows)],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # Process each scenario
    for idx, (scen, title) in enumerate(zip(scenarios, scenario_titles)):
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Get scenario data
        center = scen.get("center", {"lat": CENTER_LAT_DEFAULT, "lon": 4.0})
        center_lat, center_lon = float(center["lat"]), float(center["lon"])
        
        # Calculate symbol sizes (smaller for combined view)
        tri_nm, gap_nm = _symbol_sizes(scen)
        tri_nm *= 0.7  # Scale down for combined view
        gap_nm *= 0.8
        
        # Collect all aircraft positions for collision detection
        all_aircraft_positions = [(float(ag["lat"]), float(ag["lon"])) for ag in scen.get("agents", [])]
        
        # Add paths first
        distances = []
        for i, ag in enumerate(scen.get("agents", [])):
            color = COLORS[i % len(COLORS)]
            if "waypoint" in ag:
                lat, lon = float(ag["lat"]), float(ag["lon"])
                wp_lat, wp_lon = float(ag["waypoint"]["lat"]), float(ag["waypoint"]["lon"])
                d = latlon_distance_nm(lat, lon, wp_lat, wp_lon)
                distances.append(d)
                
                # Add path line
                fig.add_trace(go.Scatter(
                    x=[lon, wp_lon], y=[lat, wp_lat], mode="lines",
                    line=dict(width=2, dash="dash", color=color),
                    showlegend=False, hoverinfo="skip"
                ), row=row, col=col)
                
                # Add distance label (smaller for combined view)
                mlon, mlat = (lon + wp_lon) / 2.0, (lat + wp_lat) / 2.0
                fig.add_annotation(
                    x=mlon, y=mlat, text=f"{d:.1f} NM", 
                    showarrow=False, xref=f"x{idx+1}", yref=f"y{idx+1}",
                    font=dict(size=10, family="Arial, sans-serif"), 
                    bgcolor="white", bordercolor=color, borderwidth=1, opacity=0.9
                )
        
        # Add waypoints and aircraft
        for i, ag in enumerate(scen.get("agents", [])):
            color = COLORS[i % len(COLORS)]
            lat, lon, hdg = float(ag["lat"]), float(ag["lon"]), float(ag["hdg_deg"])
            
            # Add waypoint if exists
            if "waypoint" in ag:
                wp_lat, wp_lon = float(ag["waypoint"]["lat"]), float(ag["waypoint"]["lon"])
                
                # Waypoint diamond
                diamond_pts = diamond_points(wp_lat, wp_lon, max(0.6, tri_nm * 0.45))
                xs = [p[0] for p in diamond_pts]
                ys = [p[1] for p in diamond_pts]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", line=dict(width=1, color="#111"),
                    fill="toself", fillcolor=color, opacity=0.65,
                    showlegend=False, hoverinfo="skip"
                ), row=row, col=col)
                
                # Waypoint center marker
                fig.add_trace(go.Scatter(
                    x=[wp_lon], y=[wp_lat], mode="markers",
                    marker=dict(size=5, color="white", line=dict(width=1, color="#111")),
                    showlegend=False, hoverinfo="skip"
                ), row=row, col=col)
                
                # Waypoint label (smaller for combined view)
                e_nm, n_nm, xa, ya = _waypoint_label_offset(
                    center_lat, center_lon, wp_lat, wp_lon, 
                    lat, lon, gap_nm * 0.6, all_aircraft_positions
                )
                fig.add_annotation(
                    x=wp_lon + nm_to_lon(e_nm, wp_lat), y=wp_lat + nm_to_lat(n_nm),
                    text=f"<b>WP-{ag['id']}</b>", xref=f"x{idx+1}", yref=f"y{idx+1}",
                    showarrow=False, xanchor=xa, yanchor=ya,
                    bgcolor=color, bordercolor="#222", borderwidth=1, opacity=0.85,
                    font=dict(size=10, color="white", family="Arial, sans-serif")
                )
            
            # Aircraft triangle
            aircraft_pts = aircraft_triangle_points(lat, lon, hdg, tri_nm)
            xs = [p[0] for p in aircraft_pts]
            ys = [p[1] for p in aircraft_pts]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", line=dict(width=1.5, color="#111"),
                fill="toself", fillcolor=color, opacity=0.85,
                showlegend=False, hoverinfo="skip"
            ), row=row, col=col)
            
            # Aircraft heading line
            e, n = _rotate_en(0.0, tri_nm * 1.0, hdg)
            lat2 = lat + nm_to_lat(n)
            lon2 = lon + nm_to_lon(e, lat)
            fig.add_trace(go.Scatter(
                x=[lon, lon2], y=[lat, lat2], mode="lines",
                line=dict(width=2, color=color),
                showlegend=False, hoverinfo="skip"
            ), row=row, col=col)
            
            # Aircraft label (smaller for combined view)
            e_nm, n_nm, xa, ya = _label_offset(center_lat, center_lon, lat, lon, gap_nm)
            fig.add_annotation(
                x=lon + nm_to_lon(e_nm, lat), y=lat + nm_to_lat(n_nm),
                text=f"<b>{ag['id']}</b><br>HDG: {hdg:.0f}°<br>SPD: {ag['spd_kt']:.0f} kt",
                xref=f"x{idx+1}", yref=f"y{idx+1}",
                showarrow=False, xanchor=xa, yanchor=ya,
                bgcolor="white", bordercolor="#222", borderwidth=1, opacity=0.95,
                font=dict(size=11, family="Arial, sans-serif")
            )
        
        # Add center marker
        fig.add_trace(go.Scatter(
            x=[center_lon], y=[center_lat], mode="markers",
            marker=dict(size=6, color="#111"),
            showlegend=False, hoverinfo="skip"
        ), row=row, col=col)
        
        # Add radar rings (smaller radii for combined view)
        for r in [5, 10, 15]:
            xs, ys = [], []
            for k in range(0, 361, 5):
                th = math.radians(k)
                e_nm, n_nm = r * math.cos(th), r * math.sin(th)
                xs.append(center_lon + nm_to_lon(e_nm, center_lat))
                ys.append(center_lat + nm_to_lat(n_nm))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#D0D8E0", width=0.8),
                showlegend=False, hoverinfo="skip"
            ), row=row, col=col)
        
        # Set subplot viewport
        lat_min, lat_max, lon_min, lon_max = _bounds(scen)
        pad_lat = (lat_max - lat_min) * 0.2 + nm_to_lat(tri_nm * 2.0)
        pad_lon = (lon_max - lon_min) * 0.2 + nm_to_lon(tri_nm * 2.0, (lat_min + lat_max) / 2.0)
        
        fig.update_xaxes(
            range=[lon_min - pad_lon, lon_max + pad_lon],
            title_text="Longitude" if row == rows else "",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#F0F4F8",
            row=row, col=col
        )
        fig.update_yaxes(
            range=[lat_min - pad_lat, lat_max + pad_lat],
            title_text="Latitude" if col == 1 else "",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#F0F4F8",
            scaleanchor=f"x{idx+1}",
            scaleratio=1,
            row=row, col=col
        )
    
    # Update overall layout
    fig.update_layout(
        title=dict(
            text="<b style='font-size:32px'>ATC Collision Avoidance Scenarios</b><br><span style='font-size:20px'>Multi-Agent Reinforcement Learning Test Cases</span>",
            x=0.5,
            font=dict(family="Arial, sans-serif")
        ),
        template="plotly_white",
        margin=dict(l=80, r=80, t=140, b=80),
        width=1600, height=1200,
        showlegend=False,
    )
    
    # Update subplot titles with larger fonts
    fig.update_annotations(font=dict(size=16, family="Arial, sans-serif"))
    
    # Save the combined plot
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    html_path = str(out / "combined_scenarios_presentation.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    
    png_path = None
    try:
        png_path = str(out / "combined_scenarios_presentation.png")
        fig.write_image(png_path, width=2000, height=1500, scale=2.5)
    except Exception as e:
        print(f"Could not save PNG: {e}")
        png_path = None
    
    return html_path, png_path


def plot_all_scenarios_combined(scen_dir: str | Path = "scenarios", out_dir: str | Path = "plots") -> str:
    """
    Convenience function to create combined scenarios plot.
    Returns the path to the generated PNG file.
    """
    html_path, png_path = create_combined_scenarios_plot(scen_dir, out_dir)
    return png_path if png_path else html_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--file", type=str)
    g.add_argument("--scenarios-dir", type=str, default="scenarios")
    g.add_argument("--combined", action="store_true", help="Create combined plot with all scenarios")
    ap.add_argument("--out", type=str, default="plots")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    if args.file:
        print(plot_file(args.file, args.out, png=not args.no_png))
    elif args.combined:
        print(create_combined_scenarios_plot(args.scenarios_dir, args.out))
    else:
        for o in plot_dir(args.scenarios_dir, args.out, png=not args.no_png):
            print(o)
