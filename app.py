import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict, Optional


# --- 1. DOMAIN MODELS & CONSTANTS ---
CHART_SIZE = 500

BLUE_PALETTE = [
    [0.0, "#E0F2FF"],  # very low
    [0.25, "#9CCCF7"],  # low
    [0.5, "#4A90E2"],  # medium
    [0.75, "#1F78D1"],  # high
    [1.0, "#0D47A1"]  # very high
]
LIGHT_BLUE = BLUE_PALETTE[0][1]
DARK_BLUE = BLUE_PALETTE[-1][-1]

AMBER_PALETTE = [
    [0.0, "#FFF8E1"],  # very low
    [0.25, "#FFD54F"],  # low
    [0.5, "#FFB300"],  # medium
    [0.75, "#FB8C00"],  # high
    [1.0, "#E65100"]  # very high
]
LIGHT_AMBER = AMBER_PALETTE[0][1]
DARK_AMBER = AMBER_PALETTE[-1][-1]

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
    risk_df = pd.read_csv('./datasets/clean/disaster_risk_index.csv')
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

    return nat_avg_df, map_gdf, options,risk_df


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

        /* Metric label — "Monthly Spending for..." */
        .stMetric [data-testid="stMetricLabel"],
        .stMetric [data-testid="stMetricLabel"] p,
        .stMetric [data-testid="stMetricLabel"] span {
            color: #555555 !important;
            font-size: 0.9rem !important;
        }

        /* Metric value — "₱13,938.61" */
        .stMetric [data-testid="stMetricValue"],
        .stMetric [data-testid="stMetricValue"] span {
            color: #1a1a1a !important;
        }

        /* Metric delta — "₱8,946.83 vs ..." */
        .stMetric [data-testid="stMetricDelta"],
        .stMetric [data-testid="stMetricDelta"] span {
            color: #555555 !important;
        }

        /* ADD CUSTOM CSS FOR NEW SECTIONS HERE */
        </style>
    """, unsafe_allow_html=True)


def initialize_sidebar_controls(region_options: List[str]) -> Tuple[str, Optional[str], List[str]]:
    """Renders sidebar widgets and handles bulk-selection state management.
    
    Returns:
        selected_region: Primary region selection
        compare_region: Secondary region for comparison (None if not comparing)
        active_categories: List of active expenditure category column names
    """
    st.sidebar.title("🔍 Filters")

    # --- Primary Region ---
    selected_region = st.sidebar.selectbox(
        "Focus Region",
        options=region_options,
        key="primary_region"
    )

    # --- Comparison Region ---
    st.sidebar.markdown("### Compare With")
    enable_compare = st.sidebar.toggle("Enable Region Comparison", value=False)

    compare_region = None
    if enable_compare:
        # Exclude the primary region from the comparison options to avoid comparing same vs same
        compare_options = [r for r in region_options if r != selected_region]
        compare_region = st.sidebar.selectbox(
            "Compare Region",
            options=compare_options,
            key="compare_region"
        )

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

    return selected_region, compare_region, active_categories


# --- 4. VISUALIZATION ENGINE ---

def build_regional_choropleth(map_gdf: gpd.GeoDataFrame, highlight_indices: List[int]) -> go.Figure:
    """Constructs the Plotly Mapbox Choropleth."""

    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()),
        locations=map_gdf.index,
        z=map_gdf['DYNAMIC_Z'],
        colorscale=BLUE_PALETTE,
        marker_opacity=0.7,
        marker_line_width=0.5,
        marker_line_color="black",
        selectedpoints=highlight_indices,
        selected={'marker': {'opacity': 1.0}},
        unselected={'marker': {'opacity': 0.15}},
        hovertemplate="<b>%{customdata[0]}</b><br>Total: ₱%{z:,.2f}<extra></extra>",
        customdata=map_gdf[['REGION']],
    ))

    fig.update_layout(
        mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=CHART_SIZE,
    )
    return fig


def build_expenditure_bar_chart(
    data_row: pd.Series,
    categories: List[str],
    y_max: float,
    compare_row: Optional[pd.Series] = None,
    compare_label: Optional[str] = None,
    primary_label: Optional[str] = None,
) -> go.Figure:
    """Constructs the bar chart showing categorical breakdown.
    
    If compare_row is provided, renders a grouped bar chart for side-by-side comparison.
    Otherwise renders a single-region bar chart.
    """
    labels = [c.replace('_MONTHLY', '').title() for c in categories]
    fig = go.Figure()

    if compare_row is not None:
        # --- Grouped comparison mode ---
        fig.add_trace(go.Bar(
            name=primary_label or "Region A",
            x=labels,
            y=data_row[categories].values,
            marker_color=DARK_BLUE,
            text=data_row[categories].values,
            texttemplate='₱%{text:,.0f}',
            textposition='outside',
        ))
        fig.add_trace(go.Bar(
            name=compare_label or "Region B",
            x=labels,
            y=compare_row[categories].values,
            marker_color=DARK_AMBER,
            text=compare_row[categories].values,
            texttemplate='₱%{text:,.0f}',
            textposition='outside',
        ))
        fig.update_layout(barmode='group')
    else:
        # --- Single region mode (original behavior) ---
        fig.add_trace(go.Bar(
            x=labels,
            y=data_row[categories].values,
            marker_color=DARK_BLUE,
            text=data_row[categories].values,
            texttemplate='₱%{text:,.0f}',
            textposition='outside',
        ))

    fig.update_layout(
        template="plotly_white",
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        yaxis=dict(range=[0, y_max * 1.3], showticklabels=False, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=CHART_SIZE
    )
    return fig


def get_display_row(
    region: str,
    nat_avg_df: pd.DataFrame,
    map_gdf: gpd.GeoDataFrame
) -> pd.Series:
    """Helper: returns the data row for a given region name."""
    if region == "All Regions (National Avg)":
        return nat_avg_df.iloc[0]
    return map_gdf[map_gdf['REGION'] == region].iloc[0]

# [ADDED] Disaster Risk Heatmap Builder
def build_risk_heatmap(risk_df: pd.DataFrame, highlighted_region: Optional[str] = None) -> go.Figure:
    """Constructs an interactive Region × Risk Component heatmap.

    Args:
        risk_df: DataFrame from disaster_risk_index.csv
        highlighted_region: Region name to highlight in blue (None = no highlight)
    Returns:
        A Plotly Figure ready for st.plotly_chart()
    """
    # Prepare heatmap data
    heatmap_df = risk_df[[
        'PH Region',
        'Disaster Frequency (Normalized)',
        'Human Impact (Normalized)',
        'Economic Impact (Normalized)',
        'Disaster Risk Score'
    ]].copy()

    heatmap_df.columns = ['Region', 'Frequency', 'Human Impact', 'Economic Impact', 'Disaster Risk Score']
    heatmap_df = heatmap_df.set_index('Region')
    heatmap_df = heatmap_df.sort_values('Disaster Risk Score', ascending=True)

    # Build unified hover text (same tooltip regardless of which cell is hovered)
    hover_array = []
    for region in heatmap_df.index:
        region_data = risk_df[risk_df['PH Region'] == region].iloc[0]
        total_disasters = int(region_data['Disaster Count'])
        freq = heatmap_df.loc[region, 'Frequency']
        human = heatmap_df.loc[region, 'Human Impact']
        econ = heatmap_df.loc[region, 'Economic Impact']
        risk = heatmap_df.loc[region, 'Disaster Risk Score']

        hover = (
            f"<b>{region}</b><br>"
            f"<b>━━━━━━━━━━━━━━━━━━</b><br>"
            f"<b>📊 RISK ANALYSIS:</b><br>"
            f"  • <b>Risk Score: {risk:.2f}</b><br>"
            f"  • Total Disasters: {total_disasters}<br>"
            f"<b>━━━━━━━━━━━━━━━━━━</b><br>"
            f"<b>📈 COMPONENT SCORES:</b><br>"
            f"  • Frequency: {freq:.2f}<br>"
            f"  • Human Impact: {human:.2f}<br>"
            f"  • Economic Impact: {econ:.2f}<br>"
        )
        hover_array.append([hover] * len(heatmap_df.columns))

    # Base heatmap (Yellow-Orange gradient)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale=[
            [0.0, '#FFF8E1'], [0.2, '#FFE082'], [0.4, '#FFC107'],
            [0.6, '#FF9800'], [0.8, '#F57C00'], [1.0, '#E65100'],
        ],
        customdata=hover_array,
        hovertemplate='%{customdata}<extra></extra>',
        colorbar=dict(
            title="Normalized Score (0-100)",
            title_side='right',
            tickmode='linear', tick0=0, dtick=10,
            len=0.75, thickness=15,
            tickfont=dict(color='black'),
            title_font=dict(color='black')
            
        ),
        xgap=2, ygap=2
    ))

    # Blue highlighting overlay for selected region
    if highlighted_region and highlighted_region in heatmap_df.index:
        row_idx = list(heatmap_df.index).index(highlighted_region)
        for col_idx, value in enumerate(heatmap_df.loc[highlighted_region].values):
            intensity = value / 100
            if intensity < 0.2:
                color = '#E3F2FD'
            elif intensity < 0.4:
                color = '#90CAF9'
            elif intensity < 0.6:
                color = '#42A5F5'
            elif intensity < 0.8:
                color = '#1E88E5'
            else:
                color = '#1565C0'

            fig.add_shape(
                type="rect",
                x0=col_idx - 0.5, x1=col_idx + 0.5,
                y0=row_idx - 0.5, y1=row_idx + 0.5,
                fillcolor=color, opacity=0.95,
                line=dict(color="white", width=2),
                layer="above"
            )

    fig.update_layout(
        title={
            'text': '<b>Region × Risk Component Heat Map</b>',
            'x': 0.5, 'xanchor': 'center',
            'font': {'size': 18, 'color': 'black'}
        },
        xaxis=dict(title='', side='bottom', tickfont=dict(size=11, color='black'), tickangle=-45, showgrid=False),
        yaxis=dict(title='', tickfont=dict(size=10, color='black'), showgrid=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=250, r=150, t=100, b=120),
        height=700
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
    nat_avg_df, map_gdf, options_list,risk_df = fetch_and_preprocess_data()

    # Get User Inputs
    selected_region, compare_region, selected_cats = initialize_sidebar_controls(options_list)

    # Business Logic: Calculate current metric scope
    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0

    # Determine subset of data for display and map highlighting
    display_data_row = get_display_row(selected_region, nat_avg_df, map_gdf)

    # Highlight: primary region + comparison region (if any)
    if selected_region == "All Regions (National Avg)":
        indices_to_highlight = list(range(len(map_gdf)))
    else:
        indices_to_highlight = map_gdf.index[map_gdf['REGION'] == selected_region].tolist()

    compare_data_row = None
    if compare_region:
        compare_data_row = get_display_row(compare_region, nat_avg_df, map_gdf)
        # Also highlight the comparison region on the map
        if compare_region == "All Regions (National Avg)":
            indices_to_highlight = list(range(len(map_gdf)))
        else:
            compare_indices = map_gdf.index[map_gdf['REGION'] == compare_region].tolist()
            indices_to_highlight = list(set(indices_to_highlight + compare_indices))

    # Render Main UI
    st.markdown("<h1 style='text-align: left;'>🇵🇭 PH Average Monthly Expenditures</h1>", unsafe_allow_html=True)

    # KPI Section
    current_scope_total = display_data_row[selected_cats].sum() if selected_cats else 0

    if compare_region and compare_data_row is not None:
        # Side-by-side KPI metrics when comparing
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(
                label=f"Monthly Spending — {selected_region}",
                value=f"₱{current_scope_total:,.2f}"
            )
        with kpi_col2:
            compare_total = compare_data_row[selected_cats].sum() if selected_cats else 0
            delta = compare_total - current_scope_total
            if delta >= 0:
                delta_str = f"₱{delta:,.2f} vs {selected_region}"
            else:
                delta_str = f"-₱{abs(delta):,.2f} vs {selected_region}"
            st.metric(
                label=f"Monthly Spending — {compare_region}",
                value=f"₱{compare_total:,.2f}",
                delta=delta_str,
            )
    else:
        st.metric(label=f"Monthly Spending for {selected_region}", value=f"₱{current_scope_total:,.2f}")

    # Main Visuals Row: Map and Bar Chart
    col_map, col_bar = st.columns([1.7, 1.3])

    with col_map:
        fig_map = build_regional_choropleth(map_gdf, indices_to_highlight)
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    with col_bar:
        if selected_cats:
            global_max_val = map_gdf[selected_cats].max().max() if not map_gdf.empty else 10000

            # Sort categories by primary region values
            sorted_cats = display_data_row[selected_cats].sort_values(ascending=False).index.tolist()

            fig_bar = build_expenditure_bar_chart(
                data_row=display_data_row,
                categories=sorted_cats,
                y_max=global_max_val,
                compare_row=compare_data_row,
                compare_label=compare_region,
                primary_label=selected_region,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Please select at least one category in the sidebar to see the breakdown.")


    # [ADDED] Disaster Risk Heatmap Section
    st.markdown("---")
    st.subheader("🔥 Disaster Risk Heatmap")

    with st.container():
        # Map the selected_region to its PH Region name for heatmap highlighting
        # Only highlight if a specific region (not National Avg) is selected
        heatmap_highlight = None
        if selected_region != "All Regions (National Avg)":
            heatmap_highlight = selected_region

        fig_heatmap = build_risk_heatmap(risk_df, highlighted_region=heatmap_highlight)
        st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})


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