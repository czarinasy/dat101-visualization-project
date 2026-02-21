import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict


# --- 1. DOMAIN MODELS & CONSTANTS ---

class Expenditure(Enum):
    FOOD = "FOOD_MONTHLY"
    CLOTH = "CLOTH_MONTHLY"
    HOUSING_WATER = "HOUSING_WATER_MONTHLY"
    HEALTH = "HEALTH_MONTHLY"
    EDUCATION = "EDUCATION_MONTHLY"


class RegionMapping(Enum):
    """Maps FIES CSV region names to Shapefile 'name' attributes."""
    REGION_1 = ("Region I - Ilocos Region", "Region I (Ilocos Region)")
    REGION_2 = ("Region II - Cagayan Valley", "Region II (Cagayan Valley)")
    REGION_3 = ("Region III - Central Luzon", "Region III (Central Luzon)")
    REGION_4A = ("Region IVA - CALABARZON", "Region IV-A (CALABARZON)")
    MIMAROPA = ("Region IVB - MIMAROPA", "MIMAROPA Region")
    REGION_5 = ("Region V - Bicol", "Region V (Bicol Region)")
    REGION_6 = ("Region VI - Western Visayas", "Region VI (Western Visayas)")
    REGION_7 = ("Region VII - Central Visayas", "Region VII (Central Visayas)")
    REGION_8 = ("Region VIII - Eastern Visayas", "Region VIII (Eastern Visayas)")
    REGION_9 = ("Region IX - Zamboanga Peninsula", "Region IX (Zamboanga Peninsula)")
    REGION_10 = ("Region X - Northern Mindanao", "Region X (Northern Mindanao)")
    REGION_11 = ("Region XI - Davao", "Region XI (Davao Region)")
    REGION_12 = ("Region XII - SOCCSKSARGEN", "Region XII (SOCCSKSARGEN)")
    REGION_13 = ("Region XIII - Caraga", "Region XIII (Caraga)")
    NCR = ("National Capital Region", "National Capital Region (NCR)")
    CAR = ("Cordillera Administrative Region", "Cordillera Administrative Region (CAR)")
    BARMM = ("Bangsamoro Autonomous Region in Muslim Mindanao", "Bangsamoro Autonomous Region In Muslim Mindanao")

    @classmethod
    def get_map_dict(cls) -> Dict[str, str]:
        return {item.value[0]: item.value[1] for item in cls}


# --- 2. DATA SERVICES ---

@st.cache_data
def load_analytical_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame, List[str]]:
    df = pd.read_csv('./datasets/clean/fies_2023.csv')
    gdf = gpd.read_file("./datasets/raw/Regions.shp.shp")
    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    gdf = gdf.to_crs(epsg=4326)

    mapping_dict = RegionMapping.get_map_dict()
    nat_avg_df = df[df['REGION'] == "All Regions (National Avg)"].copy()
    regional_df = df[df['REGION'] != "All Regions (National Avg)"].copy()
    regional_df['MAP_NAME'] = regional_df['REGION'].map(mapping_dict)

    map_gdf = gdf.merge(regional_df, left_on='name', right_on='MAP_NAME')
    options = ["All Regions (National Avg)"] + map_gdf['REGION'].unique().tolist()

    return nat_avg_df, map_gdf, options


# --- 3. UI STYLING ---

def apply_custom_styles():
    # Set page config here to handle the Tab Name
    st.set_page_config(
        layout="wide",
        page_title="PH Average Monthly Expenditures",
        page_icon="🇵🇭"
    )

    st.markdown("""
        <style>
        button.viewerBadge_link__1S137, .main header, a.header-anchor { display: none !important; }

        .block-container { 
            padding-top: 2rem; 
            padding-bottom: 0rem; 
            max-width: 100% !important; 
            padding-left: 2rem; 
            padding-right: 2rem; 
        }

        h1 { margin-top: -30px; padding-bottom: 10px; font-size: 2rem !important; }
        .stMetric { 
            background-color: #ffffff; padding: 15px; border-radius: 8px; 
            box-shadow: 0 1px 2px rgba(0,0,0,0.1); border: 1px solid #eee; text-align: center;
        }
        div[data-testid="stCheckbox"] { margin-bottom: -15px; }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar_filters(options: List[str]) -> Tuple[str, List[str]]:
    st.sidebar.title("🔍 Filters")
    region = st.sidebar.selectbox("Focus Region", options=options)
    st.sidebar.markdown("### Categories")

    def set_category_state(state: bool):
        for e in Expenditure: st.session_state[e.name] = state

    c1, c2 = st.sidebar.columns(2)
    c1.button("Select All", on_click=set_category_state, args=(True,), use_container_width=True)
    c2.button("Clear", on_click=set_category_state, args=(False,), use_container_width=True)

    selected = []
    for e in Expenditure:
        if e.name not in st.session_state: st.session_state[e.name] = True
        if st.sidebar.checkbox(e.name.replace("_", " ").title(), key=e.name):
            selected.append(e.value)
    return region, selected


# --- 4. VISUALIZATION ENGINE ---

def create_choropleth(map_gdf: gpd.GeoDataFrame, selected_index: List[int]) -> go.Figure:
    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()), locations=map_gdf.index, z=map_gdf['DYNAMIC_Z'],
        colorscale="YlOrRd", marker_opacity=0.7, marker_line_width=0.5, marker_line_color="black",
        selectedpoints=selected_index, selected={'marker': {'opacity': 1.0}}, unselected={'marker': {'opacity': 0.15}},
        hovertemplate="<b>%{customdata[0]}</b><br>Total: ₱%{z:,.2f}<extra></extra>",
        customdata=map_gdf[['REGION']]
    ))
    fig.update_layout(
        mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500
    )
    return fig


def create_bar_chart(display_row: pd.Series, selected_cats: List[str], g_max: float) -> go.Figure:
    labels = [c.replace('_MONTHLY', '').title() for c in selected_cats]
    fig = go.Figure(go.Bar(
        x=labels, y=display_row[selected_cats].values, marker_color='#E65100',
        text=display_row[selected_cats].values, texttemplate='₱%{text:,.0f}', textposition='outside'
    ))
    fig.update_layout(
        template="plotly_white", margin={"t": 30, "b": 10, "l": 10, "r": 10},
        yaxis=dict(range=[0, g_max * 1.3], showticklabels=False, showgrid=False),
        height=500
    )
    return fig


# --- 5. MAIN ---

def main():
    apply_custom_styles()
    nat_avg_df, map_gdf, options_list = load_analytical_data()
    selected_region, selected_cats = render_sidebar_filters(options_list)

    # Calculate map Z-values
    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0

    # Determine display row and map highlight indices
    if selected_region == "All Regions (National Avg)":
        indices = list(range(len(map_gdf)))
        display_row = nat_avg_df.iloc[0]
    else:
        indices = map_gdf.index[map_gdf['REGION'] == selected_region].tolist()
        display_row = map_gdf[map_gdf['REGION'] == selected_region].iloc[0]

    st.markdown("<h1>🇵🇭 PH Average Monthly Expenditures</h1>", unsafe_allow_html=True)

    current_total = display_row[selected_cats].sum() if selected_cats else 0
    st.metric(label=f"Monthly Spending for {selected_region}", value=f"₱{current_total:,.2f}")

    col_l, col_r = st.columns([1.7, 1.3])

    with col_l:
        st.plotly_chart(create_choropleth(map_gdf, indices), use_container_width=True, config={'displayModeBar': False})
    with col_r:
        if selected_cats:
            g_max = map_gdf[selected_cats].max().max() if not map_gdf.empty else 10000
            st.plotly_chart(create_bar_chart(display_row, selected_cats, g_max), use_container_width=True,
                            config={'displayModeBar': False})
        else:
            st.info("Select categories to see breakdown.")


if __name__ == "__main__":
    main()