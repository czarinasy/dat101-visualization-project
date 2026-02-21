"""
    DASHBOARD

    To run:
        streamlit run app.py
"""

import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict


# --- 1. DOMAIN MODELS & CONSTANTS ---

class Expenditure(Enum):
    """Enumeration of expenditure categories for type-safety and consistency."""
    FOOD = "FOOD_MONTHLY"
    CLOTH = "CLOTH_MONTHLY"
    HOUSING_WATER = "HOUSING_WATER_MONTHLY"
    HEALTH = "HEALTH_MONTHLY"
    EDUCATION = "EDUCATION_MONTHLY"
    # ADD NEW CATEGORIES HERE as Enum members


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
        """Returns a dictionary mapping FIES names to Map names."""
        return {item.fies_name: item.shp_name for item in cls}

    @property
    def fies_name(self) -> str:
        """The region name as found in the CSV dataset."""
        return self.value[0]

    @property
    def shp_name(self) -> str:
        """The region name as found in the Shapefile geometry."""
        return self.value[1]


# --- 2. DATA SERVICES ---

@st.cache_data
def fetch_and_preprocess_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame, List[str]]:
    """Loads datasets and performs initial merging and coordinate transforms."""
    df = pd.read_csv('./datasets/clean/fies_2023.csv')
    gdf = gpd.read_file("./datasets/raw/Regions.shp.shp")
    # IF ADDING NEW DATA SOURCES: Load them here and return as part of the Tuple

    # Geometry simplification for performance
    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    gdf = gdf.to_crs(epsg=4326)

    # Apply mapping via Enum
    mapping_dict = RegionMapping.get_map_dict()
    nat_avg_df = df[df['REGION'] == "All Regions (National Avg)"].copy()
    regional_df = df[df['REGION'] != "All Regions (National Avg)"].copy()
    regional_df['SHP_NAME'] = regional_df['REGION'].map(mapping_dict)

    # Perform Inner Join on names
    map_gdf = gdf.merge(regional_df, left_on='name', right_on='SHP_NAME')

    # Ensure "All Regions" is always the first option in the list
    options = ["All Regions (National Avg)"] + sorted(map_gdf['REGION'].unique().tolist())

    return nat_avg_df, map_gdf, options


# --- 3. UI STYLING & FILTERS ---

def inject_custom_css():
    """Configures page layout and injects custom CSS for branding and spacing."""
    st.set_page_config(
        layout="wide",
        page_title="PH Average Monthly Expenditures",
        page_icon="🇵🇭"
    )

    st.markdown("""
        <style>
        /* Hide default Streamlit anchors and UI elements */
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

        /* ADD CUSTOM CSS FOR NEW SECTIONS HERE */
        </style>
    """, unsafe_allow_html=True)


def initialize_sidebar_controls(region_options: List[str]) -> Tuple[str, List[str]]:
    """Renders sidebar widgets and handles bulk-selection state management."""
    st.sidebar.title("🔍 Filters")
    selected_region = st.sidebar.selectbox("Focus Region", options=region_options)

    # ADD SIDEBAR SETTINGS (e.g., Theme Toggles, Date Selectors) HERE

    st.sidebar.markdown("### Categories")

    # Expenditure type check boxes
    def bulk_set_category_state(is_enabled: bool):
        for category in Expenditure:
            st.session_state[category.name] = is_enabled

    # Bulk Selection Buttons
    c1, c2 = st.sidebar.columns(2)
    c1.button("Select All", on_click=bulk_set_category_state, args=(True,), use_container_width=True)
    c2.button("Clear", on_click=bulk_set_category_state, args=(False,), use_container_width=True)

    # Individual Checkboxes
    active_categories = []
    for category in Expenditure:
        if category.name not in st.session_state:
            st.session_state[category.name] = True

        if st.sidebar.checkbox(category.name.replace("_", " ").title(), key=category.name):
            active_categories.append(category.value)

    return selected_region, active_categories


# --- 4. VISUALIZATION ENGINE ---

def build_regional_choropleth(map_gdf: gpd.GeoDataFrame, highlight_indices: List[int]) -> go.Figure:
    """Constructs the Plotly Mapbox Choropleth."""
    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()),
        locations=map_gdf.index,
        z=map_gdf['DYNAMIC_Z'],
        colorscale="YlOrRd",
        marker_opacity=0.7,
        marker_line_width=0.5,
        marker_line_color="black",
        selectedpoints=highlight_indices,
        selected={'marker': {'opacity': 1.0}},
        unselected={'marker': {'opacity': 0.15}},
        hovertemplate="<b>%{customdata[0]}</b><br>Total: ₱%{z:,.2f}<extra></extra>",
        customdata=map_gdf[['REGION']]
    ))

    fig.update_layout(
        mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500
    )
    return fig


def build_expenditure_bar_chart(data_row: pd.Series, categories: List[str], y_max: float) -> go.Figure:
    """Constructs the bar chart showing categorical breakdown."""
    labels = [c.replace('_MONTHLY', '').title() for c in categories]
    fig = go.Figure(go.Bar(
        x=labels,
        y=data_row[categories].values,
        marker_color='#E65100',
        text=data_row[categories].values,
        texttemplate='₱%{text:,.0f}',
        textposition='outside'
    ))

    fig.update_layout(
        template="plotly_white",
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        yaxis=dict(range=[0, y_max * 1.3], showticklabels=False, showgrid=False),
        height=500
    )
    return fig


# DEFINE NEW VISUALIZATION BUILDERS (e.g., build_line_chart, build_data_table) HERE

# UNCOMMENT THIS FUNCTION TO SEE EXAMPLE OF HOW TO BUILD NEW SECTION

# def build_rankings_table(map_gdf: gpd.GeoDataFrame, selected_cats: List[str]):
#     """Creates a sorted dataframe for regional comparison rankings."""
#     # Filter only relevant columns and calculate the sum
#     rank_df = map_gdf[['REGION'] + selected_cats].copy()
#     rank_df['Total Spending'] = rank_df[selected_cats].sum(axis=1)
#
#     # Sort and format for display
#     rank_df = rank_df[['REGION', 'Total Spending']].sort_values(by='Total Spending', ascending=False)
#     rank_df['Total Spending'] = rank_df['Total Spending'].map('₱{:,.2f}'.format)
#
#     return rank_df.reset_index(drop=True)


# --- 5. MAIN APPLICATION ---

def main():
    inject_custom_css()

    # Load Data
    nat_avg_df, map_gdf, options_list = fetch_and_preprocess_data()

    # Get User Inputs
    selected_region, selected_cats = initialize_sidebar_controls(options_list)

    # Business Logic: Calculate current metric scope
    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0

    # Determine subset of data for display and map highlighting
    if selected_region == "All Regions (National Avg)":
        indices_to_highlight = list(range(len(map_gdf)))
        display_data_row = nat_avg_df.iloc[0]
    else:
        indices_to_highlight = map_gdf.index[map_gdf['REGION'] == selected_region].tolist()
        display_data_row = map_gdf[map_gdf['REGION'] == selected_region].iloc[0]

    # Render Main UI
    st.markdown("<h1 style='text-align: left;'>🇵🇭 PH Average Monthly Expenditures</h1>", unsafe_allow_html=True)

    # Key Performance Indicator (KPI) Section
    # TO ADD MORE METRICS: Add more columns to the layout here
    current_scope_total = display_data_row[selected_cats].sum() if selected_cats else 0
    st.metric(label=f"Monthly Spending for {selected_region}", value=f"₱{current_scope_total:,.2f}")

    # Main Visuals Row: Map and Bar Chart
    col_map, col_bar = st.columns([1.7, 1.3])

    with col_map:
        fig_map = build_regional_choropleth(map_gdf, indices_to_highlight)
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    with col_bar:
        if selected_cats:
            global_max_val = map_gdf[selected_cats].max().max() if not map_gdf.empty else 10000
            fig_bar = build_expenditure_bar_chart(display_data_row, selected_cats, global_max_val)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Please select at least one category in the sidebar to see the breakdown.")

    # --- ADD NEW LAYOUT SECTIONS BELOW ---
    # Example: st.markdown("---")
    # Example: st.subheader("Secondary Insights")
    # Example: new_col1, new_col2 = st.columns(2)
    # -------------------------------------
    # --- ADD NEW LAYOUT SECTIONS BELOW ---

    # UNCOMMENT THIS SECTION BELOW TO SEE EXAMPLE

    # st.markdown("---")
    # st.subheader("📊 Regional Comparison Rankings")
    #
    # # Using a container to center the new section
    # with st.container():
    #     if selected_cats:
    #         # Generate the data using our new builder
    #         rankings_data = build_rankings_table(map_gdf, selected_cats)
    #
    #         # Display the table in the dashboard
    #         st.dataframe(
    #             rankings_data,
    #             use_container_width=True,
    #             hide_index=True
    #         )
    #     else:
    #         st.info("Select categories to generate rankings.")

    # -------------------------------------

if __name__ == "__main__":
    main()