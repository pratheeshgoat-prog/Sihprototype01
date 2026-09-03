import time
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString

st.set_page_config(page_title="HydroSafe", layout="wide")
st.title("🌊 HydroSafe — Flood Risk Decision Support")

# =========================================
# STUDY AREA
# =========================================
CENTER_LAT, CENTER_LON = 11.0168, 76.9558
AREA_SIZE = 0.04
lat_min, lat_max = CENTER_LAT - AREA_SIZE / 2, CENTER_LAT + AREA_SIZE / 2
lon_min, lon_max = CENTER_LON - AREA_SIZE / 2, CENTER_LON + AREA_SIZE / 2

GRID_SIZE = 150
latitudes = np.linspace(lat_min, lat_max, GRID_SIZE)
longitudes = np.linspace(lon_min, lon_max, GRID_SIZE)
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

elevation = (100 + 15 * (lat_grid - lat_min) / (lat_max - lat_min)
             + 10 * np.sin(lon_grid * 80) + 5 * np.cos(lat_grid * 80))
valley = 20 * np.exp(-((lon_grid - CENTER_LON) ** 2 + (lat_grid - CENTER_LAT) ** 2) / 0.00008)
elevation = elevation - valley

breach_lat, breach_lon = CENTER_LAT + 0.003, CENTER_LON
water_center_lat, water_center_lon = CENTER_LAT + 0.008, CENTER_LON
water_gdf = gpd.GeoDataFrame(
    {"id": ["WB001"]},
    geometry=[Point(water_center_lon, water_center_lat).buffer(0.006)],
    crs="EPSG:4326"
)

buildings = pd.DataFrame({
    "id": ["B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008"],
    "type": ["Hospital", "School", "House", "House", "Hospital", "House", "School", "House"],
    "latitude": [11.018, 11.014, 11.012, 11.020, 11.010, 11.017, 11.013, 11.022],
    "longitude": [76.955, 76.960, 76.951, 76.948, 76.963, 76.949, 76.967, 76.958]
})

roads_gdf = gpd.GeoDataFrame(
    {"road_id": ["R001", "R002", "R003"], "type": ["Major Road", "Major Road", "Local Road"]},
    geometry=[
        LineString([(76.945, 11.005), (76.955, 11.016), (76.970, 11.025)]),
        LineString([(76.948, 11.023), (76.958, 11.015), (76.970, 11.008)]),
        LineString([(76.950, 11.006), (76.960, 11.020)])
    ],
    crs="EPSG:4326"
)
if st.session_state.breach_triggered:
    if st.sidebar.button("↩ Reset to stable"):
        st.session_state.breach_triggered = False
        st.rerun()
# =========================================
# SIDEBAR — BREACH SCENARIO PARAMETERS
# (must be defined before anything below uses them)
# =========================================
st.sidebar.header("Scenario Parameters")
breach_width = st.sidebar.slider("Breach width (m)", 5, 100, 30)
breach_duration = st.sidebar.slider("Breach duration (min)", 10, 180, 60)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 50)

# =========================================
# HELPER FUNCTIONS (defined once, used everywhere below)
# =========================================
def breach_hydrograph_factor(t_min, duration):
    """Stylized outflow curve: rises, peaks, recedes. Illustrative, not real hydraulics."""
    rise_end = duration * 0.3
    plateau_end = duration * 0.7
    tail_end = duration * 2.5
    if t_min <= 0:
        return 0.0
    elif t_min < rise_end:
        return t_min / rise_end
    elif t_min < plateau_end:
        return 1.0
    elif t_min < tail_end:
        progress = (t_min - plateau_end) / (tail_end - plateau_end)
        return max(1.0 - progress * 0.8, 0.2)
    else:
        return 0.2


def simulate_flood_at_time(t_min, water_level, width, duration, rain):
    rainfall_effect = rain / 1000.0
    breach_factor = np.clip(width / 50, 0.2, 2.0)
    duration_factor = np.clip(duration / 60, 0.5, 2.0)
    hydro = breach_hydrograph_factor(t_min, duration)

    effective_water_level = (
        water_level + rainfall_effect
        + (3 * breach_factor + 2 * duration_factor) * hydro
    )

    distance = np.sqrt(((lon_grid - breach_lon) / 0.01) ** 2 + ((lat_grid - breach_lat) / 0.01) ** 2)
    attenuation = np.exp(-distance / 2.5)
    water_surface = effective_water_level * attenuation + water_level * (1 - attenuation)
    depth = np.maximum(water_surface - elevation, 0)
    return depth, depth >= 0.15


def get_depth_at_point(lat, lon, depth_grid):
    i = np.argmin(np.abs(latitudes - lat))
    j = np.argmin(np.abs(longitudes - lon))
    return depth_grid[i, j]


def classify_risk(depth):
    if depth <= 0:
        return "Safe"
    elif depth < 0.3:
        return "Low"
    elif depth < 0.6:
        return "Medium"
    elif depth < 1.2:
        return "High"
    else:
        return "Critical"


def road_max_depth(line, depth_grid, n=20):
    return max(
        get_depth_at_point(line.interpolate(t, normalized=True).y,
                            line.interpolate(t, normalized=True).x,
                            depth_grid)
        for t in np.linspace(0, 1, n)
    )


# =========================================
# DEMO INPUT DATASET: rising water level trend
# =========================================
st.subheader("📈 Dam Monitoring — Demo Sensor Feed")

if "monitoring_data" not in st.session_state:
    np.random.seed(42)
    n_readings = 30
    base_level = 105
    trend = np.linspace(0, 12, n_readings)
    noise = np.random.normal(0, 0.4, n_readings)
    st.session_state.monitoring_data = pd.DataFrame({
        "reading_no": range(n_readings),
        "hours_ago": np.linspace(29, 0, n_readings),
        "water_level_m": base_level + trend + noise
    })

monitoring_df = st.session_state.monitoring_data
BREACH_THRESHOLD = 120  # demo dam capacity limit, in meters

st.line_chart(monitoring_df.set_index("hours_ago")["water_level_m"])
st.caption(f"Demo sensor readings, synthetic. Breach threshold set at {BREACH_THRESHOLD} m for this scenario.")

# =========================================
# BREACH PREDICTION (simple linear trend extrapolation)
# =========================================
recent = monitoring_df.tail(10)
slope, intercept = np.polyfit(recent["reading_no"], recent["water_level_m"], 1)
current_level = monitoring_df["water_level_m"].iloc[-1]

if slope > 0:
    readings_to_breach = (BREACH_THRESHOLD - current_level) / slope
    hours_to_breach = max(readings_to_breach * 1, 0)  # demo assumption: 1 reading per hour
else:
    hours_to_breach = None

col1, col2, col3 = st.columns(3)
col1.metric("Current water level", f"{current_level:.1f} m")
col2.metric("Breach threshold", f"{BREACH_THRESHOLD} m")
if hours_to_breach is not None and hours_to_breach < 500:
    col3.metric("Estimated time to breach", f"{hours_to_breach:.1f} hrs")
    if hours_to_breach < 6:
        st.error(f"⚠️ CRITICAL: Breach predicted in ~{hours_to_breach:.1f} hours at current trend")
    elif hours_to_breach < 24:
        st.warning(f"⚠️ Water level rising toward breach threshold — ~{hours_to_breach:.1f} hrs")
else:
    col3.metric("Estimated time to breach", "Stable")

st.caption("Prediction method: linear trend extrapolation on recent readings — "
           "an illustrative heuristic, not a structural/geotechnical failure model.")

if "breach_triggered" not in st.session_state:
    st.session_state.breach_triggered = False

if st.button("🔴 Simulate breach now (demo trigger)"):
    st.session_state.breach_triggered = True

breach_reached = current_level >= BREACH_THRESHOLD or st.session_state.breach_triggered

# =========================================
# POST-BREACH IMPACT
# =========================================
if breach_reached:
    st.subheader("🌊 Post-Breach Impact")

    t_min = st.slider("Minutes since breach", 0, int(breach_duration * 3), 0, step=5)
    flood_depth, flooded = simulate_flood_at_time(t_min, current_level, breach_width, breach_duration, rainfall)

    buildings["flood_depth_m"] = buildings.apply(
        lambda r: get_depth_at_point(r.latitude, r.longitude, flood_depth), axis=1
    )
    buildings["risk_level"] = buildings["flood_depth_m"].apply(classify_risk)

    roads_gdf["max_flood_depth_m"] = roads_gdf.geometry.apply(lambda ln: road_max_depth(ln, flood_depth))
    roads_gdf["status"] = roads_gdf["max_flood_depth_m"].apply(
        lambda d: "Impassable" if d >= 0.3 else ("Caution" if d > 0 else "Clear")
    )

    # --- Top metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Buildings affected", int((buildings["risk_level"] != "Safe").sum()), f"/ {len(buildings)}")
    m2.metric("Critical facilities at risk",
              int(((buildings["type"].isin(["Hospital", "School"])) & (buildings["risk_level"] != "Safe")).sum()))
    m3.metric("Roads impassable", int((roads_gdf["status"] == "Impassable").sum()))
    m4.metric("Max flood depth", f"{flood_depth.max():.2f} m")

    # --- Risk category bar chart ---
    st.write("**Buildings by risk category:**")
    st.bar_chart(buildings["risk_level"].value_counts())

    # --- Map view toggle ---
    view_mode = st.radio("Map view", ["2D (reliable)", "3D Digital Twin"], horizontal=True)

    if view_mode == "2D (reliable)":
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=14)
        folium.GeoJson(water_gdf, style_function=lambda x: {"fillColor": "blue", "color": "blue", "fillOpacity": 0.4}).add_to(m)
        folium.Marker([breach_lat, breach_lon], popup="BREACH", icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

        risk_colors = {"Safe": "green", "Low": "beige", "Medium": "orange", "High": "red", "Critical": "darkred"}
        for _, row in buildings.iterrows():
            folium.CircleMarker(
                location=[row.latitude, row.longitude], radius=8,
                popup=f"{row.id} ({row.type}): {row.risk_level}, {row.flood_depth_m:.2f}m",
                color=risk_colors[row.risk_level], fill=True, fill_opacity=0.8
            ).add_to(m)

        road_colors = {"Clear": "black", "Caution": "orange", "Impassable": "red"}
        for _, row in roads_gdf.iterrows():
            folium.GeoJson(row.geometry, style_function=lambda x, c=road_colors[row.status]: {"color": c, "weight": 4}).add_to(m)

        st_folium(m, width=1000, height=550)

    else:
        terrain_df = pd.DataFrame({
            "lon": lon_grid.flatten(),
            "lat": lat_grid.flatten(),
            "elevation": elevation.flatten(),
            "depth": flood_depth.flatten()
        })
        flooded_df = terrain_df[terrain_df["depth"] > 0.15]

        terrain_layer = pdk.Layer(
            "ColumnLayer", data=terrain_df.iloc[::5],
            get_position=["lon", "lat"], get_elevation="elevation",
            elevation_scale=2, radius=15, get_fill_color=[160, 160, 160, 60],
        )
        flood_layer = pdk.Layer(
            "ColumnLayer",
            data=flooded_df.iloc[::3] if len(flooded_df) > 0 else flooded_df,
            get_position=["lon", "lat"], get_elevation="elevation + depth * 10",
            elevation_scale=2, radius=15, get_fill_color=[30, 100, 255, 180],
        )
        building_layer = pdk.Layer(
            "ScatterplotLayer", data=buildings,
            get_position=["longitude", "latitude"], get_fill_color=[255, 0, 0, 200], get_radius=25,
        )
        view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=14, pitch=45, bearing=20)
        r = pdk.Deck(layers=[terrain_layer, flood_layer, building_layer],
                     initial_view_state=view_state, map_style=None)
        st.pydeck_chart(r)

    # --- Tables ---
    st.subheader("Buildings at risk")
    st.dataframe(buildings[buildings["risk_level"] != "Safe"].sort_values("flood_depth_m", ascending=False))

    st.subheader("Road status")
    st.dataframe(roads_gdf[["road_id", "type", "max_flood_depth_m", "status"]])

else:
    st.info("Dam stable. Flood impact and 3D visualization will appear once breach threshold is reached.")
