
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString

st.set_page_config(page_title="HydroSafe", layout="wide")
st.title("🌊 HydroSafe — Flood Risk Decision Support")

# =========================================
# STUDY AREA (same as Colab)
# =========================================
CENTER_LAT, CENTER_LON = 11.0168, 76.9558
AREA_SIZE = 0.04
lat_min, lat_max = CENTER_LAT - AREA_SIZE/2, CENTER_LAT + AREA_SIZE/2
lon_min, lon_max = CENTER_LON - AREA_SIZE/2, CENTER_LON + AREA_SIZE/2

GRID_SIZE = 150
latitudes = np.linspace(lat_min, lat_max, GRID_SIZE)
longitudes = np.linspace(lon_min, lon_max, GRID_SIZE)
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

elevation = (100 + 15*(lat_grid-lat_min)/(lat_max-lat_min)
             + 10*np.sin(lon_grid*80) + 5*np.cos(lat_grid*80))
valley = 20*np.exp(-((lon_grid-CENTER_LON)**2 + (lat_grid-CENTER_LAT)**2)/0.00008)
elevation = elevation - valley

breach_lat, breach_lon = CENTER_LAT + 0.003, CENTER_LON
water_center_lat, water_center_lon = CENTER_LAT + 0.008, CENTER_LON
water_gdf = gpd.GeoDataFrame({"id": ["WB001"]}, geometry=[Point(water_center_lon, water_center_lat).buffer(0.006)], crs="EPSG:4326")

buildings = pd.DataFrame({
    "id": ["B001","B002","B003","B004","B005","B006","B007","B008"],
    "type": ["Hospital","School","House","House","Hospital","House","School","House"],
    "latitude": [11.018,11.014,11.012,11.020,11.010,11.017,11.013,11.022],
    "longitude": [76.955,76.960,76.951,76.948,76.963,76.949,76.967,76.958]
})
roads_gdf = gpd.GeoDataFrame(
    {"road_id": ["R001","R002","R003"], "type": ["Major Road","Major Road","Local Road"]},
    geometry=[
        LineString([(76.945,11.005),(76.955,11.016),(76.970,11.025)]),
        LineString([(76.948,11.023),(76.958,11.015),(76.970,11.008)]),
        LineString([(76.950,11.006),(76.960,11.020)])
    ], crs="EPSG:4326"
)

# =========================================
# SIDEBAR CONTROLS (this is the point of the dashboard)
# =========================================
st.sidebar.header("Scenario Parameters")
water_level = st.sidebar.slider("Water level (m)", 100, 130, 115)
breach_width = st.sidebar.slider("Breach width (m)", 5, 100, 30)
breach_duration = st.sidebar.slider("Breach duration (min)", 10, 180, 60)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 50)

# =========================================
# FLOOD SIMULATION
# =========================================
def simulate_flood(water_level, breach_width, breach_duration, rainfall):
    rainfall_effect = rainfall / 1000.0
    breach_factor = np.clip(breach_width/50, 0.2, 2.0)
    duration_factor = np.clip(breach_duration/60, 0.5, 2.0)
    effective_water_level = water_level + rainfall_effect + 3*breach_factor + 2*duration_factor
    distance = np.sqrt(((lon_grid-breach_lon)/0.01)**2 + ((lat_grid-breach_lat)/0.01)**2)
    attenuation = np.exp(-distance/2.5)
    water_surface = effective_water_level*attenuation + water_level*(1-attenuation)
    flood_depth = np.maximum(water_surface - elevation, 0)
    return flood_depth, flood_depth >= 0.15

flood_depth, flooded = simulate_flood(water_level, breach_width, breach_duration, rainfall)

def get_depth_at_point(lat, lon):
    i = np.argmin(np.abs(latitudes - lat))
    j = np.argmin(np.abs(longitudes - lon))
    return flood_depth[i, j]

def classify_risk(depth):
    if depth <= 0: return "Safe"
    elif depth < 0.3: return "Low"
    elif depth < 0.6: return "Medium"
    elif depth < 1.2: return "High"
    else: return "Critical"

buildings["flood_depth_m"] = buildings.apply(lambda r: get_depth_at_point(r.latitude, r.longitude), axis=1)
buildings["risk_level"] = buildings["flood_depth_m"].apply(classify_risk)

def road_max_depth(line, n=20):
    return max(get_depth_at_point(line.interpolate(t, normalized=True).y,
                                    line.interpolate(t, normalized=True).x) for t in np.linspace(0,1,n))

roads_gdf["max_flood_depth_m"] = roads_gdf.geometry.apply(road_max_depth)
roads_gdf["status"] = roads_gdf["max_flood_depth_m"].apply(
    lambda d: "Impassable" if d >= 0.3 else ("Caution" if d > 0 else "Clear"))

# =========================================
# TOP METRICS
# =========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Buildings affected", int((buildings["risk_level"] != "Safe").sum()), f"/ {len(buildings)}")
col2.metric("Critical facilities at risk",
            int(((buildings["type"].isin(["Hospital","School"])) & (buildings["risk_level"] != "Safe")).sum()))
col3.metric("Roads impassable", int((roads_gdf["status"] == "Impassable").sum()))
col4.metric("Max flood depth", f"{flood_depth.max():.2f} m")

# =========================================
# MAP
# =========================================
m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=14)
folium.GeoJson(water_gdf, style_function=lambda x: {"fillColor":"blue","color":"blue","fillOpacity":0.4}).add_to(m)
folium.Marker([breach_lat, breach_lon], popup="BREACH", icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

risk_colors = {"Safe":"green","Low":"beige","Medium":"orange","High":"red","Critical":"darkred"}
for _, row in buildings.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude], radius=8,
        popup=f"{row.id} ({row.type}): {row.risk_level}, {row.flood_depth_m:.2f}m",
        color=risk_colors[row.risk_level], fill=True, fill_opacity=0.8
    ).add_to(m)

road_colors = {"Clear":"black","Caution":"orange","Impassable":"red"}
for _, row in roads_gdf.iterrows():
    folium.GeoJson(row.geometry, style_function=lambda x, c=road_colors[row.status]: {"color": c, "weight": 4}).add_to(m)

st_folium(m, width=1000, height=550)

# =========================================
# TABLES
# =========================================
st.subheader("Buildings at risk")
st.dataframe(buildings[buildings["risk_level"] != "Safe"].sort_values("flood_depth_m", ascending=False))

st.subheader("Road status")
st.dataframe(roads_gdf[["road_id","type","max_flood_depth_m","status"]])
