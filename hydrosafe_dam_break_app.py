import time
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString
from PIL import Image

st.set_page_config(page_title="HydroSafe", layout="wide")

st.title("🌊 HydroSafe — Dam-Break Flood Animation")
st.caption(
    "Interactive DEMO simulation: the flood front expands minute-by-minute after a dam breach. "
    "This is an illustrative model, not a hydraulic/geotechnical prediction."
)

# ============================================================
# STUDY AREA / DEMO TERRAIN
# ============================================================
CENTER_LAT, CENTER_LON = 11.0168, 76.9558
AREA_SIZE = 0.04

lat_min, lat_max = CENTER_LAT - AREA_SIZE / 2, CENTER_LAT + AREA_SIZE / 2
lon_min, lon_max = CENTER_LON - AREA_SIZE / 2, CENTER_LON + AREA_SIZE / 2

GRID_SIZE = 150
latitudes = np.linspace(lat_min, lat_max, GRID_SIZE)
longitudes = np.linspace(lon_min, lon_max, GRID_SIZE)
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Synthetic terrain: higher at the edges, lower around the valley.
elevation = (
    100
    + 15 * (lat_grid - lat_min) / (lat_max - lat_min)
    + 10 * np.sin(lon_grid * 80)
    + 5 * np.cos(lat_grid * 80)
)
valley = 20 * np.exp(
    -((lon_grid - CENTER_LON) ** 2 + (lat_grid - CENTER_LAT) ** 2) / 0.00008
)
elevation = elevation - valley

# Dam / breach location.
breach_lat, breach_lon = CENTER_LAT + 0.003, CENTER_LON

# Demo water body.
water_center_lat, water_center_lon = CENTER_LAT + 0.008, CENTER_LON
water_gdf = gpd.GeoDataFrame(
    {"id": ["WB001"]},
    geometry=[Point(water_center_lon, water_center_lat).buffer(0.006)],
    crs="EPSG:4326",
)

# ============================================================
# DEMO BUILDINGS / ROADS
# ============================================================
buildings = pd.DataFrame(
    {
        "id": ["B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008"],
        "type": [
            "Hospital", "School", "House", "House",
            "Hospital", "House", "School", "House"
        ],
        "latitude": [
            11.018, 11.014, 11.012, 11.020,
            11.010, 11.017, 11.013, 11.022
        ],
        "longitude": [
            76.955, 76.960, 76.951, 76.948,
            76.963, 76.949, 76.967, 76.958
        ],
    }
)

roads_gdf = gpd.GeoDataFrame(
    {"road_id": ["R001", "R002", "R003"],
     "type": ["Major Road", "Major Road", "Local Road"]},
    geometry=[
        LineString([(76.945, 11.005), (76.955, 11.016), (76.970, 11.025)]),
        LineString([(76.948, 11.023), (76.958, 11.015), (76.970, 11.008)]),
        LineString([(76.950, 11.006), (76.960, 11.020)]),
    ],
    crs="EPSG:4326",
)

# ============================================================
# CONTROLS
# ============================================================
st.sidebar.header("⚙️ Dam-Break Scenario")

breach_width = st.sidebar.slider("Breach width (m)", 5, 100, 30)
breach_duration = st.sidebar.slider("Dam emptying time (min)", 10, 180, 60)
rainfall = st.sidebar.slider("Rainfall contribution (mm)", 0, 200, 50)

st.sidebar.divider()
st.sidebar.header("🎬 Flood Animation")

# This is the requested "time taken by the dam to affect the areas".
propagation_time = st.sidebar.slider(
    "Time for water to reach the study area (min)",
    1, 120, 30,
    help="Controls how quickly the flood front travels from the breach across the study area."
)

max_sim_time = st.sidebar.slider(
    "Animation length after breach (min)",
    30, 360, 180,
    step=10
)

animation_speed = st.sidebar.slider(
    "Animation speed",
    0.05, 1.50, 0.25,
    step=0.05,
    help="Seconds between simulated minutes. Smaller = faster."
)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "breach_triggered": True,
    "sim_minute": 0,
    "playing": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# FLOOD MODEL
# ============================================================
def breach_hydrograph_factor(t_min, duration):
    """Stylized outflow curve for visualization."""
    if t_min <= 0:
        return 0.0

    rise_end = max(duration * 0.25, 1)
    plateau_end = max(duration * 0.60, rise_end + 1)
    tail_end = max(duration * 2.5, plateau_end + 1)

    if t_min < rise_end:
        return t_min / rise_end
    if t_min < plateau_end:
        return 1.0
    if t_min < tail_end:
        progress = (t_min - plateau_end) / (tail_end - plateau_end)
        return max(1.0 - progress * 0.8, 0.15)
    return 0.15


def simulate_flood_at_time(t_min, water_level, width, duration, rain, travel_time):
    """
    Animated conceptual flood model.

    The water front expands from the breach. The front radius is controlled
    by travel_time, while depth is also influenced by terrain elevation.
    """
    rain_effect = rain / 1000.0
    breach_factor = np.clip(width / 50.0, 0.2, 2.0)
    hydro = breach_hydrograph_factor(t_min, duration)

    # Front arrival: 0 at breach, 1 after the selected propagation time.
    distance = np.sqrt(
        ((lon_grid - breach_lon) / 0.01) ** 2
        + ((lat_grid - breach_lat) / 0.01) ** 2
    )

    max_distance = max(
        np.sqrt(
            ((lon_grid - breach_lon) / 0.01) ** 2
            + ((lat_grid - breach_lat) / 0.01) ** 2
        ).max(),
        1.0,
    )

    # Convert user-selected propagation time to a moving flood-front radius.
    progress = np.clip(t_min / max(travel_time, 1), 0.0, 1.0)
    front_radius = max(0.02, progress * max_distance * 1.05)

    # Smooth edge so the water visibly advances instead of appearing instantly.
    front_width = 0.55
    front_factor = 1.0 / (
        1.0 + np.exp((distance - front_radius) * 8.0 / max(front_width, 0.1))
    )

    # More water during the breach hydrograph peak.
    hydraulic_head = (
        0.45
        + rain_effect
        + (2.8 * breach_factor + 1.8) * hydro
    )

    # Terrain-sensitive water surface. Low areas fill first.
    local_surface = (
        hydraulic_head * front_factor
        + 0.10 * hydro
        + 0.15 * (1 - elevation / np.nanmax(elevation))
    )

    # Add a directional downhill preference toward the lower portion of the valley.
    downhill_bias = np.clip(
        1.25 - 18 * (lat_grid - breach_lat), 0.45, 1.25
    )

    water_surface = (
        elevation
        + local_surface * downhill_bias * front_factor
    )

    depth = np.maximum(water_surface - elevation, 0)

    # Prevent water from appearing meaningfully outside the moving front.
    depth = np.where(distance <= front_radius + 0.35, depth, 0)

    # A small threshold makes the animated water edge visually clear.
    flooded = depth >= 0.05

    return depth, flooded, front_radius


def get_depth_at_point(lat, lon, depth_grid):
    i = np.argmin(np.abs(latitudes - lat))
    j = np.argmin(np.abs(longitudes - lon))
    return float(depth_grid[i, j])


def classify_risk(depth):
    if depth <= 0.02:
        return "Safe"
    if depth < 0.30:
        return "Low"
    if depth < 0.60:
        return "Medium"
    if depth < 1.20:
        return "High"
    return "Critical"


def road_max_depth(line, depth_grid, n=30):
    values = []
    for t in np.linspace(0, 1, n):
        pt = line.interpolate(t, normalized=True)
        values.append(get_depth_at_point(pt.y, pt.x, depth_grid))
    return max(values)


def make_water_rgba(depth):
    """
    Turn flood depth into a transparent blue raster for Folium.
    Transparent = no water; deeper water = more opaque.
    """
    d = np.clip(depth, 0, 1.5)
    alpha = np.where(d > 0.03, 70 + 150 * np.clip(d / 1.5, 0, 1), 0)

    rgba = np.zeros((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)

    # Blue/cyan water ramp.
    rgba[..., 0] = 20
    rgba[..., 1] = (150 - 60 * np.clip(d / 1.5, 0, 1)).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = alpha.astype(np.uint8)

    return rgba


# ============================================================
# CURRENT SIMULATION FRAME
# ============================================================
t_min = int(np.clip(st.session_state.sim_minute, 0, max_sim_time))

flood_depth, flooded, front_radius = simulate_flood_at_time(
    t_min,
    water_level=105,
    width=breach_width,
    duration=breach_duration,
    rain=rainfall,
    travel_time=propagation_time,
)

display_buildings = buildings.copy()
display_buildings["flood_depth_m"] = display_buildings.apply(
    lambda r: get_depth_at_point(r.latitude, r.longitude, flood_depth),
    axis=1,
)
display_buildings["risk_level"] = display_buildings["flood_depth_m"].apply(classify_risk)

display_roads = roads_gdf.copy()
display_roads["max_flood_depth_m"] = display_roads.geometry.apply(
    lambda ln: road_max_depth(ln, flood_depth)
)
display_roads["status"] = display_roads["max_flood_depth_m"].apply(
    lambda d: "Impassable" if d >= 0.30 else ("Caution" if d > 0.02 else "Clear")
)

# ============================================================
# ANIMATION CONTROLS
# ============================================================
st.subheader("🎬 Dam-Break Animation")

c1, c2, c3, c4 = st.columns(4)

if c1.button("▶️ Play / Resume", use_container_width=True):
    st.session_state.playing = True

if c2.button("⏸️ Pause", use_container_width=True):
    st.session_state.playing = False

if c3.button("🔄 Restart", use_container_width=True):
    st.session_state.sim_minute = 0
    st.session_state.playing = False
    st.rerun()

if c4.button("⏭️ +1 minute", use_container_width=True):
    st.session_state.sim_minute = min(st.session_state.sim_minute + 1, max_sim_time)
    st.session_state.playing = False
    st.rerun()

# Manual time slider.
selected_minute = st.slider(
    "⏱️ Minutes since dam breach",
    0,
    max_sim_time,
    int(st.session_state.sim_minute),
    step=1,
)

if selected_minute != st.session_state.sim_minute and not st.session_state.playing:
    st.session_state.sim_minute = selected_minute
    t_min = selected_minute
    flood_depth, flooded, front_radius = simulate_flood_at_time(
        t_min, 105, breach_width, breach_duration, rainfall, propagation_time
    )
    display_buildings["flood_depth_m"] = display_buildings.apply(
        lambda r: get_depth_at_point(r.latitude, r.longitude, flood_depth), axis=1
    )
    display_buildings["risk_level"] = display_buildings["flood_depth_m"].apply(classify_risk)
    display_roads["max_flood_depth_m"] = display_roads.geometry.apply(
        lambda ln: road_max_depth(ln, flood_depth)
    )
    display_roads["status"] = display_roads["max_flood_depth_m"].apply(
        lambda d: "Impassable" if d >= 0.30 else ("Caution" if d > 0.02 else "Clear")
    )

# ============================================================
# LIVE STATUS
# ============================================================
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("⏱ Simulation time", f"{t_min} min")
status_col2.metric(
    "🏢 Buildings affected",
    int((display_buildings["risk_level"] != "Safe").sum()),
    f"/ {len(display_buildings)}",
)
status_col3.metric(
    "🛣 Roads impassable",
    int((display_roads["status"] == "Impassable").sum()),
)
status_col4.metric("🌊 Maximum depth", f"{flood_depth.max():.2f} m")

st.progress(
    min(t_min / max(max_sim_time, 1), 1.0),
    text=f"Flood propagation: minute {t_min} / {max_sim_time}",
)

if t_min < propagation_time:
    st.info(
        f"💧 Water is still travelling outward. "
        f"Selected arrival time is {propagation_time} minutes."
    )
else:
    st.warning(
        f"🌊 Flood front has reached most of the study area. "
        f"Current simulated time: {t_min} minutes."
    )

# ============================================================
# ANIMATED 2D MAP
# ============================================================
st.subheader("🗺️ Live Flood Extent")

m = folium.Map(
    location=[CENTER_LAT, CENTER_LON],
    zoom_start=14,
    control_scale=True,
)

# Base water body.
folium.GeoJson(
    water_gdf,
    name="Reservoir",
    style_function=lambda x: {
        "fillColor": "#1976D2",
        "color": "#0D47A1",
        "weight": 2,
        "fillOpacity": 0.25,
    },
).add_to(m)

# Animated flood raster.
water_rgba = make_water_rgba(depth=flood_depth)

folium.raster_layers.ImageOverlay(
    image=water_rgba,
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.75,
    interactive=False,
    cross_origin=False,
    zindex=10,
    name="Animated Flood Water",
).add_to(m)

# Breach marker.
folium.Marker(
    [breach_lat, breach_lon],
    tooltip=f"🔴 DAM BREACH — T+{t_min} min",
    popup=(
        f"<b>Dam breach</b><br>"
        f"Simulation time: {t_min} min<br>"
        f"Breach width: {breach_width} m"
    ),
    icon=folium.Icon(color="red", icon="warning-sign"),
).add_to(m)

# Building markers change colour as water reaches them.
risk_colors = {
    "Safe": "green",
    "Low": "lightgray",
    "Medium": "orange",
    "High": "red",
    "Critical": "darkred",
}

for _, row in display_buildings.iterrows():
    depth = row["flood_depth_m"]
    risk = row["risk_level"]

    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=10 if risk != "Safe" else 7,
        tooltip=f"{row.id} — {row.type} — {risk}",
        popup=(
            f"<b>{row.id}</b><br>"
            f"Type: {row.type}<br>"
            f"Flood depth: {depth:.2f} m<br>"
            f"Risk: <b>{risk}</b><br>"
            f"Time: T+{t_min} min"
        ),
        color=risk_colors[risk],
        fill=True,
        fill_color=risk_colors[risk],
        fill_opacity=0.9,
        weight=3,
    ).add_to(m)

# Roads react to the water.
road_colors = {"Clear": "black", "Caution": "orange", "Impassable": "red"}

for _, row in display_roads.iterrows():
    status = row["status"]
    folium.GeoJson(
        row.geometry,
        tooltip=(
            f"{row.road_id}: {status} | "
            f"{row.max_flood_depth_m:.2f} m"
        ),
        style_function=lambda feature, c=road_colors[status]: {
            "color": c,
            "weight": 6,
            "opacity": 0.9,
        },
    ).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=1200, height=620, key=f"flood-map-{t_min}")

# ============================================================
# IMPACT TABLES
# ============================================================
st.subheader("🏢 Buildings at Risk")

affected = display_buildings[
    display_buildings["risk_level"] != "Safe"
].sort_values("flood_depth_m", ascending=False)

if affected.empty:
    st.success("No buildings are flooded at this simulated minute.")
else:
    st.dataframe(
        affected[
            ["id", "type", "latitude", "longitude", "flood_depth_m", "risk_level"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("🛣️ Road Status")
st.dataframe(
    display_roads[
        ["road_id", "type", "max_flood_depth_m", "status"]
    ],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# AUTO-ADVANCE — ONE SIMULATED MINUTE PER FRAME
# ============================================================
if st.session_state.playing:
    if st.session_state.sim_minute >= max_sim_time:
        st.session_state.playing = False
        st.success("✅ Animation reached the end of the selected simulation time.")
    else:
        time.sleep(animation_speed)
        st.session_state.sim_minute += 1
        st.rerun()

st.caption(
    "⚠️ DEMO MODEL: the terrain, buildings, roads and flood propagation are synthetic. "
    "For real emergency planning, replace the synthetic DEM and assets with validated "
    "hydraulic/hydrologic data and a calibrated dam-break model."
)
