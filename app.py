
import math
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HydroSafe — Tehri Dam",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# TEHRI DAM
# ============================================================

TEHRI_LAT = 30.3778
TEHRI_LON = 78.4806


# ============================================================
# TITLE
# ============================================================

st.title("🌊 HydroSafe — Tehri Dam AI Flood Digital Twin")

st.caption(
    "Fixed demonstration scenario for Tehri Dam, Uttarakhand. "
    "The flood prediction shown here is a demonstration model "
    "and is not an emergency-grade hydraulic forecast."
)


# ============================================================
# SESSION STATE
# ============================================================

if "simulation_time" not in st.session_state:
    st.session_state.simulation_time = 0.0

if "playing" not in st.session_state:
    st.session_state.playing = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🤖 AI Flood Digital Twin",
        "⚙️ Dam-Break Simulator"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("🏔️ Tehri Dam — 3D Flood Digital Twin")

    st.info(
        "The fixed demo is centred on Tehri Dam, Uttarakhand."
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("🎬 3D Animation Controls")

    max_time = st.sidebar.slider(
        "Simulation duration (minutes)",
        min_value=30,
        max_value=240,
        value=120,
        step=10
    )

    animation_step = st.sidebar.slider(
        "Simulation step (minutes)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

    animation_speed = st.sidebar.slider(
        "Animation speed",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
        step=0.05
    )

    travel_time = st.sidebar.slider(
        "Flood arrival time (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        if st.button("▶️ PLAY", use_container_width=True):
            st.session_state.playing = True

    with col_b:
        if st.button("⏸️ PAUSE", use_container_width=True):
            st.session_state.playing = False

    with col_c:
        if st.button("🔄 RESTART", use_container_width=True):
            st.session_state.simulation_time = 0.0
            st.session_state.playing = False
            st.rerun()

    with col_d:
        if st.button("+1 MIN", use_container_width=True):
            st.session_state.simulation_time = min(
                max_time,
                st.session_state.simulation_time + 1.0
            )

    # --------------------------------------------------------
    # TIME SLIDER
    # --------------------------------------------------------

    selected_time = st.slider(
        "⏱️ Time after dam break",
        min_value=0.0,
        max_value=float(max_time),
        value=float(
            min(
                st.session_state.simulation_time,
                max_time
            )
        ),
        step=0.1
    )

    if not st.session_state.playing:
        st.session_state.simulation_time = selected_time

    simulation_time = st.session_state.simulation_time

    # --------------------------------------------------------
    # TERRAIN
    # --------------------------------------------------------

    radius_km = 8
    grid_size = 65

    lat_range = np.linspace(
        TEHRI_LAT - radius_km / 111.0,
        TEHRI_LAT + radius_km / 111.0,
        grid_size
    )

    lon_scale = 111.0 * math.cos(
        math.radians(TEHRI_LAT)
    )

    lon_range = np.linspace(
        TEHRI_LON - radius_km / lon_scale,
        TEHRI_LON + radius_km / lon_scale,
        grid_size
    )

    lon_grid, lat_grid = np.meshgrid(
        lon_range,
        lat_range
    )

    x = (
        lon_grid - TEHRI_LON
    ) * 92.0

    y = (
        lat_grid - TEHRI_LAT
    ) * 111.0

    elevation = (
        900.0
        + 150.0 * np.exp(
            -(
                (x + 2.0) ** 2 / 20.0
                + (y - 2.0) ** 2 / 30.0
            )
        )
        + 120.0 * np.exp(
            -(
                (x - 3.0) ** 2 / 15.0
                + (y + 3.0) ** 2 / 25.0
            )
        )
        - 180.0 * np.exp(
            -(
                x ** 2 / 5.0
                + (y + 1.0) ** 2 / 40.0
            )
        )
        + 20.0 * np.sin(x / 2.0)
        + 15.0 * np.cos(y / 3.0)
    )

    terrain = pd.DataFrame(
        {
            "lon": lon_grid.ravel(),
            "lat": lat_grid.ravel(),
            "elevation": elevation.ravel()
        }
    )

    # --------------------------------------------------------
    # FLOOD MODEL
    # --------------------------------------------------------

    distance = np.sqrt(
        (
            (lon_grid - TEHRI_LON) * 92.0
        ) ** 2
        +
        (
            (lat_grid - TEHRI_LAT) * 111.0
        ) ** 2
    )

    flood_speed = (
        radius_km / max(travel_time, 1)
    )

    front_radius = (
        flood_speed * simulation_time
    )

    flood_front = 1.0 / (
        1.0
        +
        np.exp(
            (distance - front_radius) / 0.8
        )
    )

    direction = (
        0.65 * (-y)
        +
        0.35 * (-x)
    )

    direction = (
        direction - direction.min()
    ) / (
        direction.max()
        - direction.min()
        + 1e-9
    )

    depth = (
        8.0
        * flood_front
        * (0.55 + 0.45 * direction)
        * np.exp(-distance / 12.0)
    )

    depth = np.maximum(
        depth,
        0.0
    )

    terrain["flood_depth"] = depth.ravel()

    terrain["water_top"] = (
        terrain["elevation"]
        + terrain["flood_depth"]
    )

    # --------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------

    buildings = pd.DataFrame(
        {
            "name": [
                "Tehri Dam",
                "Tehri Town",
                "New Tehri",
                "Hospital",
                "School",
                "Residential Area 1",
                "Residential Area 2",
                "Road Junction"
            ],
            "type": [
                "Dam",
                "Town",
                "Town",
                "Hospital",
                "School",
                "Residential",
                "Residential",
                "Road"
            ],
            "lat": [
                TEHRI_LAT,
                30.375,
                30.390,
                30.382,
                30.386,
                30.365,
                30.350,
                30.370
            ],
            "lon": [
                TEHRI_LON,
                78.480,
                78.495,
                78.465,
                78.470,
                78.465,
                78.450,
                78.475
            ]
        }
    )

    # --------------------------------------------------------
    # LOCATION FLOOD DEPTH
    # --------------------------------------------------------

    def get_depth(lat, lon):

        d = math.sqrt(
            (
                (lon - TEHRI_LON) * 92.0
            ) ** 2
            +
            (
                (lat - TEHRI_LAT) * 111.0
            ) ** 2
        )

        front = (
            flood_speed
            * simulation_time
        )

        arrival = 1.0 / (
            1.0
            + math.exp(
                (d - front) / 0.8
            )
        )

        return (
            8.0
            * arrival
            * math.exp(-d / 12.0)
        )

    buildings["flood_depth_m"] = [
        get_depth(lat, lon)
        for lat, lon in zip(
            buildings["lat"],
            buildings["lon"]
        )
    ]

    def classify(depth_value):

        if depth_value < 0.05:
            return "SAFE"

        if depth_value < 0.5:
            return "WATCH"

        if depth_value < 1.5:
            return "HIGH"

        return "CRITICAL"

    buildings["risk"] = (
        buildings["flood_depth_m"]
        .apply(classify)
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    affected = int(
        (
            buildings["risk"] != "SAFE"
        ).sum()
    )

    critical = int(
        (
            buildings["risk"] == "CRITICAL"
        ).sum()
    )

    max_depth = float(
        depth.max()
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Simulation Time",
            f"{simulation_time:.1f} min"
        )

    with m2:
        st.metric(
            "Affected Locations",
            affected
        )

    with m3:
        st.metric(
            "Critical Locations",
            critical
        )

    with m4:
        st.metric(
            "Maximum Flood Depth",
            f"{max_depth:.2f} m"
        )

    # --------------------------------------------------------
    # 3D TERRAIN
    # --------------------------------------------------------

    terrain_display = terrain.iloc[::4].copy()

    terrain_layer = pdk.Layer(
        "ColumnLayer",
        data=terrain_display,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=2,
        radius=18,
        get_fill_color=[110, 110, 110, 150],
        pickable=True
    )

    # --------------------------------------------------------
    # 3D WATER
    # --------------------------------------------------------

    water = terrain_display[
        terrain_display["flood_depth"] > 0.05
    ].copy()

    water_layer = pdk.Layer(
        "ColumnLayer",
        data=water,
        get_position=["lon", "lat"],
        get_elevation="water_top",
        elevation_scale=2,
        radius=20,
        get_fill_color=[20, 130, 255, 190],
        pickable=True
    )

    # --------------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------------

    buildings["height"] = (
        30.0
        + buildings["flood_depth_m"] * 15.0
    )

    def risk_color(risk):

        if risk == "SAFE":
            return [50, 180, 80, 240]

        if risk == "WATCH":
            return [255, 210, 50, 240]

        if risk == "HIGH":
            return [255, 120, 30, 245]

        return [230, 20, 20, 250]

    buildings["color"] = (
        buildings["risk"]
        .apply(risk_color)
    )

    building_layer = pdk.Layer(
        "ColumnLayer",
        data=buildings,
        get_position=["lon", "lat"],
        get_elevation="height",
        elevation_scale=1,
        radius=45,
        get_fill_color="color",
        pickable=True
    )

    # --------------------------------------------------------
    # DAM
    # --------------------------------------------------------

    dam = pd.DataFrame(
        {
            "lon": [TEHRI_LON],
            "lat": [TEHRI_LAT],
            "height": [180],
            "name": ["TEHRI DAM"]
        }
    )

    dam_layer = pdk.Layer(
        "ColumnLayer",
        data=dam,
        get_position=["lon", "lat"],
        get_elevation="height",
        radius=100,
        get_fill_color=[210, 30, 30, 255],
        pickable=True
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = buildings.copy()

    labels["label"] = (
        labels["name"]
        + " | "
        + labels["risk"]
        + " | "
        + labels["flood_depth_m"]
        .round(2)
        .astype(str)
        + " m"
    )

    label_layer = pdk.Layer(
        "TextLayer",
        data=labels,
        get_position=["lon", "lat"],
        get_text="label",
        get_size=14,
        get_color=[255, 255, 255, 255],
        billboard=True
    )

    # --------------------------------------------------------
    # 3D CAMERA
    # --------------------------------------------------------

    view = pdk.ViewState(
        latitude=TEHRI_LAT,
        longitude=TEHRI_LON,
        zoom=11.8,
        pitch=60,
        bearing=20
    )

    deck = pdk.Deck(
        layers=[
            terrain_layer,
            water_layer,
            building_layer,
            dam_layer,
            label_layer
        ],
        initial_view_state=view,
        tooltip={
            "html":
                "<b>{name}</b><br/>"
                "Risk: {risk}<br/>"
                "Flood depth: {flood_depth_m} m",
            "style": {
                "backgroundColor": "black",
                "color": "white"
            }
        }
    )

    st.pydeck_chart(
        deck,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "📍 Live Affected Locations"
    )

    st.dataframe(
        buildings[
            [
                "name",
                "type",
                "lat",
                "lon",
                "flood_depth_m",
                "risk"
            ]
        ].sort_values(
            "flood_depth_m",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ANIMATION
    # --------------------------------------------------------

    if (
        st.session_state.playing
        and st.session_state.simulation_time < max_time
    ):

        time.sleep(
            animation_speed
        )

        st.session_state.simulation_time = min(
            float(max_time),
            st.session_state.simulation_time
            + animation_step
        )

        st.rerun()

    elif (
        st.session_state.playing
        and st.session_state.simulation_time >= max_time
    ):

        st.session_state.playing = False


# ============================================================
# TAB 2 — CUSTOM SIMULATOR
# ============================================================

with tab2:

    st.header(
        "⚙️ Custom Dam-Break Simulator"
    )

    st.write(
        "Modify the scenario inputs and run the demonstration "
        "AI-assisted flood-risk calculation."
    )

    left, right = st.columns(2)

    with left:

        latitude = st.number_input(
            "Dam Latitude",
            value=float(TEHRI_LAT),
            format="%.6f"
        )

        longitude = st.number_input(
            "Dam Longitude",
            value=float(TEHRI_LON),
            format="%.6f"
        )

        rainfall = st.slider(
            "Rainfall (mm)",
            0,
            500,
            100
        )

        water_level = st.slider(
            "Reservoir Water Level (m)",
            0,
            300,
            200
        )

    with right:

        breach_width = st.slider(
            "Breach Width (m)",
            5,
            300,
            50
        )

        breach_duration = st.slider(
            "Breach Formation Time (min)",
            1,
            180,
            30
        )

        prediction_time = st.slider(
            "Prediction Time (min)",
            1,
            360,
            60
        )

        rainfall_intensity = st.slider(
            "Rainfall Intensity",
            0.0,
            3.0,
            1.0,
            0.1
        )

    st.divider()

    if st.button(
        "🤖 RUN AI FLOOD PREDICTION",
        type="primary",
        use_container_width=True
    ):

        risk_score = (
            0.20 * min(
                rainfall / 500.0,
                1.0
            )
            +
            0.25 * min(
                water_level / 300.0,
                1.0
            )
            +
            0.25 * min(
                breach_width / 300.0,
                1.0
            )
            +
            0.15 * min(
                breach_duration / 180.0,
                1.0
            )
            +
            0.15 * min(
                prediction_time / 360.0,
                1.0
            )
        )

        risk_score *= (
            0.8
            +
            0.2 * rainfall_intensity
        )

        risk_score = min(
            risk_score,
            1.0
        )

        if risk_score < 0.25:
            prediction = "LOW"

        elif risk_score < 0.50:
            prediction = "MODERATE"

        elif risk_score < 0.75:
            prediction = "HIGH"

        else:
            prediction = "CRITICAL"

        st.session_state.prediction = {
            "score": risk_score,
            "prediction": prediction,
            "lat": latitude,
            "lon": longitude
        }

    if st.session_state.prediction is not None:

        result = (
            st.session_state.prediction
        )

        st.subheader(
            "🤖 AI Prediction Result"
        )

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Predicted Flood Risk",
                result["prediction"]
            )

        with r2:
            st.metric(
                "AI Risk Score",
                f"{result['score'] * 100:.1f}%"
            )

        st.info(
            "Prediction centre: "
            f"{result['lat']:.6f}, "
            f"{result['lon']:.6f}"
        )

        st.warning(
            "Demonstration model only. "
            "Real-world deployment requires calibrated "
            "DEM, hydraulic modelling, validated flood data, "
            "and authoritative dam information."
        )
