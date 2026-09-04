
import io
import math
import zipfile
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import streamlit.components.v1 as components

from shapely.geometry import Point, Polygon, LineString

try:
    from sklearn.ensemble import RandomForestRegressor
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HydroSafe X — Tehri Dam Digital Twin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 HydroSafe X")
st.caption(
    "Tehri Dam Digital Twin • Physics-Informed Flood Modelling • "
    "AI Surrogate • 3D Visualization • Emergency Decision Support"
)

st.warning(
    "DEMO / RESEARCH PROTOTYPE — This model is not an operational "
    "flood warning system. Real deployment requires validated DEM, "
    "bathymetry, dam geometry, calibrated hydraulic parameters, "
    "authoritative infrastructure data and field validation."
)


# ============================================================
# TEHRI DAM
# ============================================================

TEHRI_LAT = 30.37778
TEHRI_LON = 78.48056

EARTH_RADIUS = 6371000.0


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "rainfall": 75.0,
    "water_level": 825.0,
    "breach_width": 80.0,
    "breach_duration": 30.0,
    "simulation_time": 60,
    "grid_size": 90,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEO FUNCTIONS
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * EARTH_RADIUS
        * math.asin(math.sqrt(a))
        / 1000
    )


# ============================================================
# TERRAIN
# ============================================================

@st.cache_data
def create_terrain(
    center_lat,
    center_lon,
    grid_size
):

    span_km = 7

    x = np.linspace(
        -span_km * 1000,
        span_km * 1000,
        grid_size
    )

    y = np.linspace(
        -span_km * 1000,
        span_km * 1000,
        grid_size
    )

    xx, yy = np.meshgrid(x, y)

    # Himalayan-looking terrain.
    mountains = (
        300
        + 180 * np.sin(xx / 1500)
        + 130 * np.cos(yy / 1800)
        + 100 * np.sin((xx + yy) / 2200)
    )

    valley = (
        850
        * np.exp(
            -(
                (yy / 2600) ** 2
                +
                (xx / 9000) ** 2
            )
        )
    )

    river = (
        240
        * np.exp(
            -(
                (yy + 0.28 * xx)
                / 650
            ) ** 2
        )
    )

    elevation = (
        mountains
        - valley
        - river
    )

    elevation -= elevation.min()
    elevation += 650

    lat = (
        center_lat
        +
        yy / 111320
    )

    lon = (
        center_lon
        +
        xx
        /
        (
            111320
            * math.cos(
                math.radians(center_lat)
            )
        )
    )

    return (
        xx,
        yy,
        elevation,
        lat,
        lon
    )


# ============================================================
# BUILDINGS
# ============================================================

@st.cache_data
def create_buildings(
    center_lat,
    center_lon,
    count=2500
):

    rng = np.random.default_rng(2026)

    latitudes = (
        center_lat
        -
        np.abs(
            rng.normal(
                0.012,
                0.020,
                count
            )
        )
    )

    longitudes = (
        center_lon
        +
        rng.normal(
            0,
            0.026,
            count
        )
    )

    building_types = rng.choice(
        [
            "Residential",
            "School",
            "Hospital",
            "Commercial",
            "Government",
            "Industrial"
        ],
        count,
        p=[
            0.62,
            0.06,
            0.025,
            0.17,
            0.06,
            0.065
        ]
    )

    population = rng.integers(
        2,
        15,
        count
    )

    return gpd.GeoDataFrame(
        {
            "id": [
                f"B-{i:05d}"
                for i in range(1, count + 1)
            ],
            "type": building_types,
            "population": population,
            "critical": [
                x in [
                    "School",
                    "Hospital",
                    "Government"
                ]
                for x in building_types
            ]
        },
        geometry=[
            Point(
                lon,
                lat
            )
            for lat, lon
            in zip(
                latitudes,
                longitudes
            )
        ],
        crs="EPSG:4326"
    )


# ============================================================
# ROADS
# ============================================================

@st.cache_data
def create_roads(
    center_lat,
    center_lon
):

    rows = []

    for i in range(-6, 7):

        y = (
            center_lat
            +
            i * 0.006
        )

        coords = [
            (
                center_lon - 0.055,
                y
            ),
            (
                center_lon - 0.025,
                y + 0.003
            ),
            (
                center_lon + 0.005,
                y - 0.003
            ),
            (
                center_lon + 0.032,
                y + 0.004
            ),
            (
                center_lon + 0.060,
                y
            )
        ]

        rows.append(
            {
                "road_id":
                    f"R-{len(rows)+1:03d}",
                "name":
                    f"Primary Corridor {len(rows)+1}",
                "class":
                    "Primary",
                "geometry":
                    LineString(coords)
            }
        )

    return gpd.GeoDataFrame(
        rows,
        crs="EPSG:4326"
    )


# ============================================================
# AREAS
# ============================================================

@st.cache_data
def create_areas(
    center_lat,
    center_lon
):

    definitions = [
        (
            "Tehri Reservoir Sector",
            0.004,
            0.000,
            0.010
        ),
        (
            "New Tehri Sector",
            -0.010,
            0.012,
            0.012
        ),
        (
            "Bhagirathi Corridor",
            -0.024,
            -0.006,
            0.013
        ),
        (
            "Downstream Settlement",
            -0.040,
            0.018,
            0.015
        ),
        (
            "Tehri Garhwal Sector",
            -0.058,
            -0.020,
            0.018
        ),
        (
            "Emergency Planning Zone",
            -0.078,
            0.006,
            0.020
        )
    ]

    rows = []

    for i, (
        name,
        dlat,
        dlon,
        radius
    ) in enumerate(definitions):

        lat = (
            center_lat
            + dlat
        )

        lon = (
            center_lon
            + dlon
        )

        coords = []

        for angle in np.linspace(
            0,
            2 * np.pi,
            32
        ):

            coords.append(
                (
                    lon
                    +
                    radius
                    * math.cos(angle),

                    lat
                    +
                    radius
                    * math.sin(angle)
                )
            )

        rows.append(
            {
                "area_id":
                    f"A-{i+1:03d}",
                "area_name":
                    name,
                "geometry":
                    Polygon(coords)
            }
        )

    return gpd.GeoDataFrame(
        rows,
        crs="EPSG:4326"
    )


# ============================================================
# 2D PHYSICS-INSPIRED FLOOD MODEL
# ============================================================

def flood_model(
    elevation,
    x,
    y,
    rainfall,
    water_level,
    breach_width,
    breach_duration,
    simulation_time
):

    h = np.zeros_like(
        elevation,
        dtype=float
    )

    n = elevation.shape[0]

    dx = max(
        abs(
            x[1] - x[0]
        ),
        1
    )

    # Initial reservoir.
    reservoir = (
        (x.reshape(1, -1) < 0)
        &
        (
            np.abs(
                y.reshape(-1, 1)
            ) < 2500
        )
    )

    head = np.maximum(
        water_level
        -
        elevation,
        0
    )

    h = np.where(
        reservoir,
        np.minimum(
            head,
            10
        ),
        0
    )

    frames = {}

    frame_times = np.linspace(
        0,
        simulation_time,
        min(
            13,
            simulation_time + 1
        ),
        dtype=int
    )

    steps = min(
        700,
        max(
            60,
            simulation_time * 5
        )
    )

    dt = (
        simulation_time
        * 60
        / steps
    )

    g = 9.81

    rainfall_source = (
        rainfall
        / 1000
        / max(
            simulation_time * 60,
            1
        )
    )

    breach_cells = max(
        1,
        int(
            breach_width
            / dx
        )
    )

    mid = n // 2

    for step in range(
        steps + 1
    ):

        minutes = (
            step
            * dt
            / 60
        )

        breach_fraction = min(
            1,
            minutes
            /
            max(
                breach_duration,
                1
            )
        )

        release = (
            breach_fraction
            *
            (
                0.8
                +
                breach_width
                / 100
                +
                max(
                    water_level - 800,
                    0
                )
                / 100
            )
        )

        source = np.zeros_like(
            h
        )

        left = max(
            1,
            mid
            -
            breach_cells // 2
        )

        right = min(
            n - 1,
            mid
            +
            breach_cells // 2
        )

        source[
            mid + 1:,
            left:right + 1
        ] = (
            release
            * dt
            * 0.015
        )

        source += (
            rainfall_source
            * dt
            * 0.35
        )

        h += source

        # Terrain gradient.
        dzdx = (
            np.gradient(
                elevation + h,
                axis=1
            )
            / dx
        )

        dzdy = (
            np.gradient(
                elevation + h,
                axis=0
            )
            / dx
        )

        acceleration = (
            -g
            * (
                dzdx
                +
                dzdy
            )
            * 0.0008
        )

        h += (
            acceleration
            * dt
            * 0.08
        )

        # Diffusion.
        neighbours = (
            np.roll(h, 1, axis=0)
            +
            np.roll(h, -1, axis=0)
            +
            np.roll(h, 1, axis=1)
            +
            np.roll(h, -1, axis=1)
        ) / 4

        h = (
            h * 0.92
            +
            neighbours * 0.08
        )

        h = np.maximum(
            h,
            0
        )

        nearest = int(
            frame_times[
                np.argmin(
                    np.abs(
                        frame_times
                        -
                        minutes
                    )
                )
            ]
        )

        frames[nearest] = h.copy()

    return h, frames


# ============================================================
# AI SURROGATE MODEL
# ============================================================

@st.cache_resource
def train_ai_model():

    if not ML_AVAILABLE:
        return None

    rng = np.random.default_rng(
        42
    )

    n = 7000

    rainfall = rng.uniform(
        0,
        300,
        n
    )

    water_level = rng.uniform(
        770,
        850,
        n
    )

    breach = rng.uniform(
        10,
        300,
        n
    )

    duration = rng.uniform(
        5,
        180,
        n
    )

    distance = rng.uniform(
        0.1,
        12,
        n
    )

    terrain = rng.uniform(
        650,
        1600,
        n
    )

    # Synthetic training target for prototype.
    target = (
        np.maximum(
            0,
            1
            -
            distance
            /
            (
                2
                +
                rainfall * 0.006
                +
                breach * 0.008
                +
                np.maximum(
                    water_level - 800,
                    0
                ) * 0.015
            )
        )
        *
        (
            2
            +
            rainfall / 90
            +
            breach / 100
            +
            np.maximum(
                water_level - 800,
                0
            ) / 30
        )
        *
        (
            1
            +
            np.maximum(
                900 - terrain,
                0
            ) / 1000
        )
    )

    X = np.column_stack(
        [
            rainfall,
            water_level,
            breach,
            duration,
            distance,
            terrain
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        target
    )

    return model


# ============================================================
# BUILDING IMPACT
# ============================================================

def calculate_building_impacts(
    buildings,
    rainfall,
    water_level,
    breach_width,
    breach_duration
):

    b = buildings.copy()

    distances = np.array(
        [
            distance_km(
                TEHRI_LAT,
                TEHRI_LON,
                geom.y,
                geom.x
            )
            for geom in b.geometry
        ]
    )

    radius = (
        2.5
        +
        rainfall * 0.006
        +
        breach_width * 0.010
        +
        max(
            water_level - 800,
            0
        )
        * 0.018
        +
        breach_duration * 0.005
    )

    physics_depth = (
        np.maximum(
            0,
            1
            -
            distances
            /
            max(
                radius,
                0.1
            )
        )
        *
        (
            2
            +
            rainfall / 90
            +
            breach_width / 100
            +
            max(
                water_level - 800,
                0
            ) / 25
        )
    )

    if ML_AVAILABLE:

        model = train_ai_model()

        X = np.column_stack(
            [
                np.full(
                    len(b),
                    rainfall
                ),
                np.full(
                    len(b),
                    water_level
                ),
                np.full(
                    len(b),
                    breach_width
                ),
                np.full(
                    len(b),
                    breach_duration
                ),
                distances,
                np.full(
                    len(b),
                    850
                )
            ]
        )

        ai_depth = model.predict(
            X
        )

        depth = (
            physics_depth * 0.7
            +
            ai_depth * 0.3
        )

    else:

        depth = physics_depth

    arrival = np.where(
        depth > 0.25,
        np.maximum(
            1,
            distances * 7
        ),
        np.inf
    )

    risk_score = (
        np.clip(
            depth / 8,
            0,
            1
        )
        * 0.65
        +
        np.clip(
            (
                100
                -
                arrival
            ) / 100,
            0,
            1
        )
        * 0.20
        +
        np.clip(
            rainfall / 300,
            0,
            1
        )
        * 0.15
    )

    risk = np.select(
        [
            risk_score >= 0.70,
            risk_score >= 0.40
        ],
        [
            "HIGH",
            "MEDIUM"
        ],
        default="LOW"
    )

    b["depth_m"] = depth
    b["arrival_min"] = arrival
    b["risk_score"] = risk_score
    b["risk"] = risk
    b["affected"] = (
        depth >= 0.25
    )
    b["ai_probability"] = np.clip(
        risk_score * 0.92 + 0.04,
        0,
        1
    )

    return b


# ============================================================
# AREA IMPACT
# ============================================================

def calculate_area_impacts(
    areas,
    rainfall,
    water_level,
    breach_width,
    breach_duration
):

    a = areas.copy()

    values = []

    radius = (
        2.5
        +
        rainfall * 0.006
        +
        breach_width * 0.010
        +
        max(
            water_level - 800,
            0
        )
        * 0.018
        +
        breach_duration * 0.005
    )

    for geom in a.geometry:

        center = geom.centroid

        d = distance_km(
            TEHRI_LAT,
            TEHRI_LON,
            center.y,
            center.x
        )

        depth = (
            max(
                0,
                1
                -
                d / max(
                    radius,
                    0.1
                )
            )
            *
            (
                3
                +
                rainfall / 100
                +
                breach_width / 100
            )
        )

        values.append(
            depth
        )

    a["depth_m"] = values

    a["risk"] = np.select(
        [
            a["depth_m"] >= 5,
            a["depth_m"] >= 2
        ],
        [
            "HIGH",
            "MEDIUM"
        ],
        default="LOW"
    )

    a["affected"] = (
        a["depth_m"] >= 0.25
    )

    return a


# ============================================================
# 3D DIGITAL TWIN
# ============================================================

def render_3d_twin(
    breach_width,
    breach_duration
):

    html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

html,body {{
    margin:0;
    padding:0;
    overflow:hidden;
    background:#071421;
    font-family:Arial;
}}

#hud {{
    position:absolute;
    z-index:10;
    top:15px;
    left:15px;
    color:white;
    background:rgba(0,0,0,.68);
    padding:15px;
    border-radius:12px;
    line-height:1.5;
}}

canvas {{
    display:block;
}}

</style>

</head>

<body>

<div id="hud">

<b>TEHRI DAM — LIVE DIGITAL TWIN</b>

<br>

Breach width:
{breach_width:.0f} m

<br>

Breach formation:
{breach_duration:.0f} min

<br>

<span id="status">
Initializing...
</span>

</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>

<script>

const scene = new THREE.Scene();

scene.background = new THREE.Color(
        0x071421
    );

const camera = new THREE.PerspectiveCamera(
        45,
        innerWidth / innerHeight,
        0.1,
        3000
    );

camera.position.set(
    28,
    18,
    32
);

camera.lookAt(
    0,
    5,
    0
);

const renderer = new THREE.WebGLRenderer(
        {{
            antialias:true
        }}
    );

renderer.setSize(
    innerWidth,
    innerHeight
);

renderer.setPixelRatio(
    Math.min(
        devicePixelRatio,
        2
    )
);

document.body.appendChild(
    renderer.domElement
);

scene.add(
    new THREE.AmbientLight(
        0xffffff,
        1.2
    )
);

const light = new THREE.DirectionalLight(
        0xffffff,
        2
    );

light.position.set(
    25,
    35,
    25
);

scene.add(
    light
);


// TERRAIN

const terrainGeometry = new THREE.PlaneGeometry(
        65,
        48,
        55,
        40
    );

const positions = terrainGeometry.attributes.position;

for (
    let i = 0;
    i < positions.count;
    i++
) {{

    const x = positions.getX(i);

    const y = positions.getY(i);

    const h = 2
        +
        3.5 *
        Math.sin(
            x * .15
        )
        +
        2.2 *
        Math.cos(
            y * .20
        )
        +
        1.5 *
        Math.sin(
            (x+y)*.25
        );

    positions.setZ(
        i,
        h
    );
}}

terrainGeometry.computeVertexNormals();

const terrain = new THREE.Mesh(
        terrainGeometry,
        new THREE.MeshStandardMaterial(
            {{
                color:0x4d7154,
                roughness:1
            }}
        )
    );

terrain.rotation.x = -Math.PI / 2;

terrain.position.y = -1;

scene.add(
    terrain
);


// DAM

const damWidth = 19;
const damHeight = 11;

const damBlocks = [];

for (
    let i=0;
    i<24;
    i++
) {{

    const block = new THREE.Mesh(
            new THREE.BoxGeometry(
                damWidth / 24 - .06,
                damHeight,
                2.8
            ),
            new THREE.MeshStandardMaterial(
                {{
                    color:0x9b9ea2,
                    roughness:.85
                }}
            )
        );

    block.position.set(
        -damWidth/2
        +
        (i+.5)
        *
        damWidth/24,
        damHeight/2,
        0
    );

    scene.add(
        block
    );

    damBlocks.push(
        block
    );
}}


// RESERVOIR

const reservoir = new THREE.Mesh(
        new THREE.BoxGeometry(
            31,
            5.2,
            20
        ),
        new THREE.MeshStandardMaterial(
            {{
                color:0x128fd0,
                transparent:true,
                opacity:.68
            }}
        )
    );

reservoir.position.set(
    -15,
    1.5,
    0
);

scene.add(
    reservoir
);


// FLOOD

const flood = new THREE.Mesh(
        new THREE.PlaneGeometry(
            5,
            19
        ),
        new THREE.MeshStandardMaterial(
            {{
                color:0x17b8ed,
                transparent:true,
                opacity:.72,
                side:THREE.DoubleSide
            }}
        )
    );

flood.rotation.x = -Math.PI / 2;

flood.position.set(
    3,
    .15,
    0
);

scene.add(
    flood
);


// BUILDINGS

const buildings = new THREE.Group();

for (
    let i=0;
    i<90;
    i++
) {{

    const height = .7
        +
        Math.random()
        * 4.5;

    const building = new THREE.Mesh(
            new THREE.BoxGeometry(
                .62,
                height,
                .62
            ),
            new THREE.MeshStandardMaterial(
                {{
                    color:0xd6c8a8
                }}
            )
        );

    building.position.set(
        5
        +
        Math.random()
        * 27,
        height/2,
        -9
        +
        Math.random()
        * 18
    );

    buildings.add(
        building
    );
}}

scene.add(
    buildings
);


// WATER PARTICLES

const particleCount = 1000;

const particleGeometry = new THREE.BufferGeometry();

const particlePositions = new Float32Array(
        particleCount * 3
    );

for (
    let i=0;
    i<particleCount;
    i++
) {{

    particlePositions[
        i*3
    ] = Math.random()
        * 38;

    particlePositions[
        i*3+1
    ] = .25
        +
        Math.random()
        * .25;

    particlePositions[
        i*3+2
    ] = -9
        +
        Math.random()
        * 18;
}}

particleGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(
        particlePositions,
        3
    )
);

const particles = new THREE.Points(
        particleGeometry,
        new THREE.PointsMaterial(
            {{
                color:0x70dcff,
                size:.11
            }}
        )
    );

scene.add(
    particles
);


// ANIMATION

let elapsed = 0;

const clock = new THREE.Clock();

function animate() {{

    requestAnimationFrame(
        animate
    );

    elapsed += clock.getDelta();

    const cycle = 18;

    const t = (
            elapsed % cycle
        ) / cycle;

    const breakProgress = Math.min(
            1,
            Math.max(
                0,
                (t-.12)/.30
            )
        );

    const floodProgress = Math.min(
            1,
            Math.max(
                0,
                (t-.28)/.72
            )
        );

    const center = 12;

    damBlocks.forEach(
        (block,i) => {{

            const d = Math.abs(
                    i-center
                );

            const local = Math.max(
                    0,
                    breakProgress
                    -
                    d*.045
                );

            block.position.y = damHeight/2
                -
                local*7;

            block.rotation.z = local
                *
                (
                    i<center
                    ? -.65
                    : .65
                );

        }}
    );

    flood.scale.x = 1
        +
        floodProgress
        * 11;

    flood.position.x = 3
        +
        floodProgress
        * 20;


    const p = particleGeometry
        .attributes
        .position
        .array;

    for (
        let i=0;
        i<particleCount;
        i++
    ) {{

        p[i*3] += .045
            +
            floodProgress
            * .12;

        if (
            p[i*3] > 38
        )
            p[i*3] = -1;

        p[i*3+1] = .25
            +
            Math.sin(
                elapsed*3+i
            )*.04;
    }}

    particleGeometry
        .attributes
        .position
        .needsUpdate=true;


    document.getElementById(
        "status"
    ).innerText = "Dam break: "
        +
        Math.round(
            breakProgress*100
        )
        +
        "% | Flood: "
        +
        Math.round(
            floodProgress*100
        )
        +
        "%";


    renderer.render(
        scene,
        camera
    );
}}

animate();


window.addEventListener(
    "resize",
    () => {{

        camera.aspect = innerWidth
            /
            innerHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            innerWidth,
            innerHeight
        );

    }}
);

</script>

</body>
</html>
"""

    components.html(
        html,
        height=650,
        scrolling=False
    )


# ============================================================
# 3D GEO MAP
# ============================================================

def render_geo_map(
    buildings,
    areas
):

    b = buildings.copy()

    # WebGL protection for large datasets.
    if len(b) > 3000:

        b = b.sample(
            3000,
            random_state=42
        )

    b["lat"] = (
        b.geometry.y
    )

    b["lon"] = (
        b.geometry.x
    )

    b["height"] = np.clip(
        b["depth_m"] * 4 + 4,
        4,
        35
    )

    building_layer = pdk.Layer(
            "ColumnLayer",
            data=b,
            get_position= "[lon,lat]",
            get_elevation= "height",
            elevation_scale=1,
            radius=28,
            pickable=True,
            auto_highlight=True
        )

    dam_df = pd.DataFrame(
        [
            {
                "lat":TEHRI_LAT,
                "lon":TEHRI_LON,
                "name":
                    "TEHRI DAM / BREACH"
            }
        ]
    )

    dam_layer = pdk.Layer(
            "ScatterplotLayer",
            data=dam_df,
            get_position= "[lon,lat]",
            get_radius=450,
            pickable=True
        )

    area_data = []

    for _, row in areas.iterrows():

        area_data.append(
            {
                "polygon":[
                    [x,y]
                    for x,y
                    in row.geometry.exterior.coords
                ],
                "name":
                    row["area_name"],
                "risk":
                    row["risk"],
                "depth":
                    round(
                        float(
                            row["depth_m"]
                        ),
                        2
                    )
            }
        )

    area_layer = pdk.Layer(
            "PolygonLayer",
            data=area_data,
            get_polygon= "polygon",
            get_elevation=20,
            extruded=True,
            filled=True,
            opacity=.30,
            pickable=True
        )

    deck = pdk.Deck(
            layers=[
                building_layer,
                dam_layer,
                area_layer
            ],
            initial_view_state= pdk.ViewState(
                    latitude= TEHRI_LAT,
                    longitude= TEHRI_LON,
                    zoom=11.2,
                    pitch=58,
                    bearing=15
                ),
            tooltip={
                "html":
                    "<b>{name}</b>"
                    "<br/>Risk: {risk}"
                    "<br/>Depth: {depth} m"
                    "<br/>Building depth: {depth_m} m"
            }
        )

    st.pydeck_chart(
        deck,
        use_container_width=True
    )


# ============================================================
# SHAPEFILE EXPORT
# ============================================================

def shapefile_zip(
    gdf
):

    gdf = gdf.copy()

    rename = {}
    used = set()

    for column in gdf.columns:

        if column == "geometry":
            continue

        name = str(
            column
        ).upper()[:10]

        original = name
        counter = 1

        while name in used:

            suffix = str(
                counter
            )

            name = (
                original[
                    :10-len(suffix)
                ]
                +
                suffix
            )

            counter += 1

        used.add(name)
        rename[column] = name

    gdf = gdf.rename(
        columns=rename
    )

    with tempfile.TemporaryDirectory() as td:

        output = (
            Path(td)
            /
            "hydrosafe_output.shp"
        )

        gdf.to_file(
            output,
            driver= "ESRI Shapefile",
            encoding="UTF-8"
        )

        buffer = io.BytesIO()

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as archive:

            for file in Path(td).glob(
                "hydrosafe_output.*"
            ):

                archive.write(
                    file,
                    file.name
                )

        buffer.seek(0)

        return buffer.getvalue()


# ============================================================
# KML EXPORT
# ============================================================

def kml_export(
    gdf
):

    root = ET.Element(
        "kml",
        {
            "xmlns":
                "http://www.opengis.net/kml/2.2"
        }
    )

    document = ET.SubElement(
        root,
        "Document"
    )

    for _, row in gdf.iterrows():

        placemark = ET.SubElement(
                document,
                "Placemark"
            )

        name = ET.SubElement(
                placemark,
                "name"
            )

        name.text = str(
            row.get(
                "name",
                row.get(
                    "area_name",
                    "HydroSafe Feature"
                )
            )
        )

        geometry = row.geometry

        if geometry.geom_type == "Point":

            point = ET.SubElement(
                    placemark,
                    "Point"
                )

            coordinates = ET.SubElement(
                    point,
                    "coordinates"
                )

            coordinates.text = f"{geometry.x},{geometry.y},0"

        elif (
            geometry.geom_type
            == "LineString"
        ):

            line = ET.SubElement(
                    placemark,
                    "LineString"
                )

            coordinates = ET.SubElement(
                    line,
                    "coordinates"
                )

            coordinates.text = " ".join(
                f"{x},{y},0"
                for x,y
                in geometry.coords
            )

        elif (
            geometry.geom_type
            == "Polygon"
        ):

            polygon = ET.SubElement(
                    placemark,
                    "Polygon"
                )

            outer = ET.SubElement(
                    polygon,
                    "outerBoundaryIs"
                )

            ring = ET.SubElement(
                    outer,
                    "LinearRing"
                )

            coordinates = ET.SubElement(
                    ring,
                    "coordinates"
                )

            coordinates.text = " ".join(
                f"{x},{y},0"
                for x,y
                in geometry.exterior.coords
            )

    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🎛️ Scenario Controls"
    )

    st.markdown(
        "### 🇮🇳 Fixed Demo"
    )

    st.code(
        f"Tehri Dam\n"
        f"Latitude: {TEHRI_LAT}\n"
        f"Longitude: {TEHRI_LON}"
    )

    st.session_state[
        "rainfall"
    ] = st.slider(
        "🌧️ Rainfall (mm)",
        0,
        300,
        int(
            st.session_state[
                "rainfall"
            ]
        )
    )

    st.session_state[
        "water_level"
    ] = st.slider(
        "💧 Water Level (m)",
        760.0,
        850.0,
        float(
            st.session_state[
                "water_level"
            ]
        )
    )

    st.session_state[
        "breach_width"
    ] = st.slider(
        "💥 Breach Width (m)",
        10.0,
        300.0,
        float(
            st.session_state[
                "breach_width"
            ]
        )
    )

    st.session_state[
        "breach_duration"
    ] = st.slider(
        "⏱️ Breach Formation (min)",
        1.0,
        180.0,
        float(
            st.session_state[
                "breach_duration"
            ]
        )
    )

    st.session_state[
        "simulation_time"
    ] = st.slider(
        "🕒 Simulation Time (min)",
        5,
        180,
        int(
            st.session_state[
                "simulation_time"
            ]
        )
    )

    st.session_state[
        "grid_size"
    ] = st.select_slider(
        "🧮 Physics Grid",
        options=[
            50,
            70,
            90,
            110,
            130
        ],
        value=int(
            st.session_state[
                "grid_size"
            ]
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "Preparing digital twin data..."
):

    (
        xx,
        yy,
        elevation,
        lat_grid,
        lon_grid
    ) = create_terrain(
        TEHRI_LAT,
        TEHRI_LON,
        st.session_state[
            "grid_size"
        ]
    )

    buildings = create_buildings(
            TEHRI_LAT,
            TEHRI_LON
        )

    roads = create_roads(
            TEHRI_LAT,
            TEHRI_LON
        )

    areas = create_areas(
            TEHRI_LAT,
            TEHRI_LON
        )


# ============================================================
# RUN MODEL
# ============================================================

with st.spinner(
    "Running physics-informed flood simulation..."
):

    final_depth, frames = flood_model(
            elevation,
            xx[0, :],
            yy[:, 0],
            st.session_state[
                "rainfall"
            ],
            st.session_state[
                "water_level"
            ],
            st.session_state[
                "breach_width"
            ],
            st.session_state[
                "breach_duration"
            ],
            st.session_state[
                "simulation_time"
            ]
        )


# ============================================================
# IMPACT
# ============================================================

affected_buildings = calculate_building_impacts(
        buildings,
        st.session_state[
            "rainfall"
        ],
        st.session_state[
            "water_level"
        ],
        st.session_state[
            "breach_width"
        ],
        st.session_state[
            "breach_duration"
        ]
    )

affected_areas = calculate_area_impacts(
        areas,
        st.session_state[
            "rainfall"
        ],
        st.session_state[
            "water_level"
        ],
        st.session_state[
            "breach_width"
        ],
        st.session_state[
            "breach_duration"
        ]
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🤖 AI + 3D DIGITAL TWIN",
        "⚙️ MODELLING LAB",
        "🚨 EMERGENCY DECISION",
        "📤 GIS OUTPUT"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.subheader(
        "🇮🇳 Tehri Dam — Live 3D Digital Twin"
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    total_buildings = len(
            affected_buildings
        )

    flooded = int(
            affected_buildings[
                "affected"
            ].sum()
        )

    high = int(
            (
                affected_buildings[
                    "risk"
                ]
                == "HIGH"
            ).sum()
        )

    population = int(
            affected_buildings[
                affected_buildings[
                    "affected"
                ]
            ]["population"].sum()
        )

    max_depth = float(
            affected_buildings[
                "depth_m"
            ].max()
        )

    c1.metric(
        "Buildings",
        f"{total_buildings:,}"
    )

    c2.metric(
        "Affected",
        f"{flooded:,}"
    )

    c3.metric(
        "High Risk",
        f"{high:,}"
    )

    c4.metric(
        "Population Exposed",
        f"{population:,}"
    )

    c5.metric(
        "Max Depth",
        f"{max_depth:.2f} m"
    )

    st.markdown(
        "### 🎬 Continuous Structural Break Animation"
    )

    render_3d_twin(
        st.session_state[
            "breach_width"
        ],
        st.session_state[
            "breach_duration"
        ]
    )

    st.markdown(
        "### 🗺️ 3D Geographic Impact"
    )

    render_geo_map(
        affected_buildings,
        affected_areas
    )

    st.markdown(
        "### 🧠 AI Building Risk Ranking"
    )

    st.dataframe(
        affected_buildings[
            [
                "id",
                "type",
                "population",
                "critical",
                "depth_m",
                "arrival_min",
                "risk",
                "ai_probability"
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2
# ============================================================

with tab2:

    st.subheader(
        "⚙️ Modelling Input / Output Framework"
    )

    left,right = st.columns(2)

    with left:

        st.markdown(
            "### Input Parameters"
        )

        st.write(
            f"Dam latitude: "
            f"**{TEHRI_LAT:.6f}**"
        )

        st.write(
            f"Dam longitude: "
            f"**{TEHRI_LON:.6f}**"
        )

        st.write(
            f"Rainfall: "
            f"**{st.session_state['rainfall']} mm**"
        )

        st.write(
            f"Water level: "
            f"**{st.session_state['water_level']} m**"
        )

        st.write(
            f"Breach width: "
            f"**{st.session_state['breach_width']} m**"
        )

        st.write(
            f"Breach duration: "
            f"**{st.session_state['breach_duration']} min**"
        )

        st.write(
            f"Simulation time: "
            f"**{st.session_state['simulation_time']} min**"
        )

    with right:

        st.markdown(
            "### Modelling Pipeline"
        )

        st.success(
            "01 • Terrain generation"
        )

        st.success(
            "02 • Breach source"
        )

        st.success(
            "03 • Flood propagation"
        )

        st.success(
            "04 • AI surrogate"
        )

        st.success(
            "05 • Building impact"
        )

        st.success(
            "06 • Emergency prioritization"
        )

    st.markdown(
        "### 🌊 Flood Depth Grid"
    )

    step = max(
        1,
        elevation.shape[0]
        // 60
    )

    flood_points = []

    for i in range(
        0,
        elevation.shape[0],
        step
    ):

        for j in range(
            0,
            elevation.shape[1],
            step
        ):

            depth = float(
                    final_depth[
                        i,j
                    ]
                )

            if depth > 0.05:

                flood_points.append(
                    {
                        "lat":
                            float(
                                lat_grid[
                                    i,j
                                ]
                            ),
                        "lon":
                            float(
                                lon_grid[
                                    i,j
                                ]
                            ),
                        "depth_m":
                            depth
                    }
                )

    if flood_points:

        flood_df = pd.DataFrame(
                flood_points
            )

        st.map(
            flood_df,
            latitude="lat",
            longitude="lon"
        )

        st.metric(
            "Flood Cells > 5 cm",
            f"{len(flood_df):,}"
        )

    else:

        st.info(
            "No flood cells above display threshold."
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:

    st.subheader(
        "🚨 Emergency Decision Support"
    )

    st.caption(
        "Prototype emergency prioritization layer."
    )

    priority = affected_buildings[
            affected_buildings[
                "affected"
            ]
        ].copy()

    priority[
        "priority_score"
    ] = (
        priority[
            "risk_score"
        ] * 0.55
        +
        priority[
            "ai_probability"
        ] * 0.25
        +
        priority[
            "critical"
        ].astype(float)
        * 0.20
    )

    priority = priority.sort_values(
            "priority_score",
            ascending=False
        )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Critical Facilities",
        int(
            priority[
                "critical"
            ].sum()
        )
    )

    c2.metric(
        "Estimated Exposed Population",
        int(
            priority[
                "population"
            ].sum()
        )
    )

    c3.metric(
        "Priority Locations",
        min(
            len(priority),
            100
        )
    )

    st.markdown(
        "### 🛟 Priority Evacuation Locations"
    )

    if len(priority):

        st.dataframe(
            priority[
                [
                    "id",
                    "type",
                    "population",
                    "critical",
                    "depth_m",
                    "arrival_min",
                    "risk",
                    "priority_score"
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No locations exceed the prototype threshold."
        )

    st.markdown(
        "### 🛣️ Road Impact"
    )

    road_rows = []

    for _, road in roads.iterrows():

        intersects = affected_areas[
                affected_areas.geometry.intersects(
                    road.geometry
                )
            ]

        road_rows.append(
            {
                "road_id":
                    road["road_id"],
                "name":
                    road["name"],
                "class":
                    road["class"],
                "affected":
                    bool(
                        len(intersects)
                        and
                        intersects[
                            "affected"
                        ].any()
                    )
            }
        )

    road_df = pd.DataFrame(
            road_rows
        )

    st.dataframe(
        road_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Production upgrade: connect this layer to a "
        "validated road graph and shelter database to "
        "calculate safest evacuation routes."
    )


# ============================================================
# TAB 4
# ============================================================

with tab4:

    st.subheader(
        "📤 GIS Export Centre"
    )

    export_type = st.selectbox(
            "Choose dataset",
            [
                "Affected Buildings",
                "Affected Areas",
                "Road Network",
                "All Buildings"
            ]
        )

    if export_type == "Affected Buildings":

        export_gdf = affected_buildings[
                affected_buildings[
                    "affected"
                ]
            ]

    elif export_type == "Affected Areas":

        export_gdf = affected_areas[
                affected_areas[
                    "affected"
                ]
            ]

    elif export_type == "Road Network":

        export_gdf = roads

    else:

        export_gdf = affected_buildings

    st.metric(
        "Features",
        f"{len(export_gdf):,}"
    )

    if len(export_gdf):

        try:

            shp_data = shapefile_zip(
                    export_gdf
                )

            kml_data = kml_export(
                    export_gdf
                )

            c1,c2 = st.columns(2)

            with c1:

                st.download_button(
                    "⬇️ Download SHP",
                    data=shp_data,
                    file_name= "hydrosafe_output.zip",
                    mime= "application/zip",
                    use_container_width=True
                )

            with c2:

                st.download_button(
                    "🌍 Download KML",
                    data=kml_data,
                    file_name= "hydrosafe_output.kml",
                    mime= "application/vnd.google-earth.kml+xml",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Export error: {error}"
            )

    st.markdown(
        "### 📦 Large Dataset Capability"
    )

    st.write(
        f"""
        Buildings: **{len(buildings):,}**

        Roads: **{len(roads):,}**

        Planning areas: **{len(areas):,}**

        The application keeps the analytical GeoDataFrames
        separate from the WebGL visualization layer. Large
        datasets are sampled only for visualization while
        complete datasets remain available for GIS export.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HydroSafe X • SIH Top-End Prototype • "
    "Physics-Informed Flood Modelling • AI • "
    "3D Digital Twin • GIS • Emergency Decision Support"
)
