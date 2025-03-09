# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import graphviz
import pydeck as pdk
from shapely import Point
import leafmap.foliumap as leafmap
import leafmap.colormaps as cm

import movingpandas as mpd

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -5px;
        margin-bottom: -5px;
        margin-left: -5px;
        margin-right: -5px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title(":shark: Shark Tracker w/ MovingPandas")
st.divider()

def color_categ(gdf, categ, palette):
    if gdf[categ].dtype == 'object' or gdf[categ].nunique() < 10:
        palette = sns.color_palette(palette, n_colors=gdf[categ].nunique())
        value_to_color = {val: mcolors.to_hex(palette[i]) for i, val in enumerate(gdf[categ].unique())}
        gdf['color'] = gdf[categ].map(value_to_color)
    else:
        cmap = mcolors.ListedColormap(sns.color_palette(palette, n_colors=256))
        norm = mcolors.Normalize(vmin=gdf[categ].min(), vmax=gdf[categ].max())
        gdf['color'] = gdf[categ].apply(lambda x: mcolors.to_hex(cmap(norm(x))))
    
    gdf['color'] = gdf['color'].fillna('#000000')
    return gdf['color']



@st.cache_data()
def load_data():
    sharks = pd.read_csv(path_data + '\\seanoe_sharks.csv', delimiter=';')
    sharks = sharks.drop_duplicates(subset=['Shark', 'DT'])
    sharks['Shark'] = sharks['Shark'].str.replace(r'(\d+)$', lambda x: x.group(1).zfill(2), regex=True)
    sharks['DT'] = pd.to_datetime(sharks['DT'], dayfirst=True)
    sharks = gpd.GeoDataFrame(sharks, geometry=gpd.points_from_xy(sharks.Longitude, sharks.Latitude, crs="EPSG:4326"), crs="EPSG:4326")
    sharks = sharks.set_index('DT')

    sharks_traj = mpd.TrajectoryCollection(sharks, traj_id_col='Shark')
    sharks_traj.add_speed(overwrite=True,units=("km", "h"))
    sharks_traj.add_distance(overwrite=True,units="km")
    sharks_traj.add_timedelta(overwrite=True)


    sharks_pings_gdf = sharks_traj.to_point_gdf()
    sharks_pings_gdf['geometry'] = sharks_pings_gdf['geometry'].buffer(0.0001)
    sharks_pings_gdf['speed'] = sharks_pings_gdf.speed*5

    sharks_traj_gdf = sharks_traj.to_traj_gdf()
    sharks_traj_gdf['geometry'] = sharks_traj_gdf['geometry'].buffer(0.0001)

    sharks_traj_gdf_simp = mpd.MinTimeDeltaGeneralizer(sharks_traj).generalize(tolerance=dt.timedelta(minutes=30))
    sharks_traj_gdf_simp = sharks_traj_gdf_simp.to_traj_gdf()
    sharks_traj_gdf_simp['geometry'] = sharks_traj_gdf_simp['geometry'].buffer(0.0001)

    sharks_traj_gdf_smooth = mpd.KalmanSmootherCV(sharks_traj).smooth()
    sharks_traj_gdf_smooth = sharks_traj_gdf_smooth.to_traj_gdf()
    sharks_traj_gdf_smooth['geometry'] = sharks_traj_gdf_smooth['geometry'].buffer(0.0001)

    sharks_traj_gdf_area = []
    for i in sharks_traj.to_traj_gdf().Shark.unique():
        sharks_traj_gdf_area.append({'Shark' : i, 'geometry' : sharks_traj.get_trajectory(i).get_mcp()})
    sharks_traj_gdf_area = pd.DataFrame(sharks_traj_gdf_area)
    sharks_traj_gdf_area = gpd.GeoDataFrame(sharks_traj_gdf_area, geometry=sharks_traj_gdf_area.geometry, crs="EPSG:4326")

    sharks_traj_gdf_hotspots = mpd.TrajectoryCollectionAggregator(sharks_traj, 100, 0, 1).get_clusters_gdf()
    sharks_traj_gdf_hotspots['lat'] = sharks_traj_gdf_hotspots.geometry.y
    sharks_traj_gdf_hotspots['lon'] = sharks_traj_gdf_hotspots.geometry.x

    return sharks, sharks_traj, sharks_pings_gdf, sharks_traj_gdf, sharks_traj_gdf_simp, sharks_traj_gdf_smooth, sharks_traj_gdf_area, sharks_traj_gdf_hotspots
sharks, sharks_traj, sharks_pings_gdf, sharks_traj_gdf, sharks_traj_gdf_simp, sharks_traj_gdf_smooth, sharks_traj_gdf_area, sharks_traj_gdf_hotspots = load_data()


with st.sidebar:
    st.logo(path_cda + '\\0_Branding\\logo_large.png', size='large')
    sel_sharks = st.segmented_control('White Sharks Selection', sharks_traj_gdf.Shark.unique(), selection_mode='multi')

#
#

shark_palette = 'deep'

if len(sel_sharks) == 0:
    st.warning('Select at least one shark.')
else:
    cols = st.columns(2)
    with cols[0]:
        # Table
        st.write('Shark Statistics')
        df = sharks_pings_gdf[sharks_pings_gdf.Shark.isin(sel_sharks)].groupby(['Shark', 'Sex', 'Length'])[['distance', 'speed', 'timedelta']].agg({'distance' : 'sum', 'speed' : 'mean', 'timedelta' : 'sum'}).reset_index()
        df['speed'] = df.speed.round(1)
        df['distance'] = df.distance.round(1)
        df = df.rename(columns={'distance' : 'Total Distance Traveled (km)', 'speed' : 'Average Speed (km/h)', 'timedelta' : 'Time Elapsed'})
        st.dataframe(df)

        # Speed plot
        st.write('Shark Speed Distribution')
        df = sharks_pings_gdf[sharks_pings_gdf.Shark.isin(sel_sharks) & (sharks_pings_gdf.speed <= 70)].copy()
        df['color'] = color_categ(df, 'Shark', shark_palette)
        fig = px.histogram(df, x="speed", color='Shark', color_discrete_sequence=df['color'].unique(), barmode="overlay", histnorm="percent")
        st.plotly_chart(fig)
    with cols[1]:
        # Map
        m = leafmap.Map(zoom=4)
        # Shark Pings
        sharks_pings_gdf_sel = sharks_pings_gdf[sharks_pings_gdf.Shark.isin(sel_sharks)].drop(columns=['timedelta'])
        sharks_pings_gdf_sel['color'] = color_categ(sharks_pings_gdf_sel, 'Shark', shark_palette)
        m.add_gdf(
            sharks_pings_gdf_sel,
            layer_name='Shark Pings',
            style_function=lambda feature: {
                "color": feature['properties']['color'],
                "fillOpacity": 0.8
            }
        )

        # Shark Paths
        sharks_traj_gdf_sel = sharks_traj_gdf[sharks_traj_gdf.Shark.isin(sel_sharks)].copy()
        sharks_traj_gdf_sel['color'] = color_categ(sharks_traj_gdf_sel, 'Shark', shark_palette)
        m.add_gdf(
            sharks_traj_gdf_sel,
            layer_name='Shark Paths',
            style_function=lambda feature: {
                "color": feature['properties']['color'],
                "fillOpacity": 0.8
            }
        )

        # Shark Paths Simplified
        sharks_traj_gdf_simp = sharks_traj_gdf_simp[sharks_traj_gdf_simp.Shark.isin(sel_sharks)].copy()
        sharks_traj_gdf_simp['color'] = color_categ(sharks_traj_gdf_simp, 'Shark', shark_palette)
        m.add_gdf(
            sharks_traj_gdf_simp,
            layer_name='Shark Paths Simplified',
            style_function=lambda feature: {
                "color": feature['properties']['color'],
                "fillOpacity": 0.8
            }
        )

        # Shark Paths Areas
        sharks_traj_gdf_area = sharks_traj_gdf_area[sharks_traj_gdf_area.Shark.isin(sel_sharks)].copy()
        sharks_traj_gdf_area['color'] = color_categ(sharks_traj_gdf_area, 'Shark', shark_palette)
        m.add_gdf(
            sharks_traj_gdf_area,
            layer_name='Shark Paths Areas',
            style_function=lambda feature: {
                "color": feature['properties']['color'],
                "fillOpacity": 0.8
            }
        )

        # Shark Paths Smoothed
        sharks_traj_gdf_smooth = sharks_traj_gdf_smooth[sharks_traj_gdf_smooth.Shark.isin(sel_sharks)].copy()
        sharks_traj_gdf_smooth['color'] = color_categ(sharks_traj_gdf_smooth, 'Shark', shark_palette)
        m.add_gdf(
            sharks_traj_gdf_smooth,
            layer_name='Shark Paths Smoothed',
            style_function=lambda feature: {
                "color": feature['properties']['color'],
                "fillOpacity": 0.8
            }
        )

        # Shark Hot Spots
        m.add_heatmap(sharks_traj_gdf_hotspots, name='Shark Hot Spots', latitude='lat', longitude='lon', value='n', radius=10, blur=15)

        m.add_legend(legend_dict=dict(zip(sharks_traj_gdf_sel['Shark'], sharks_traj_gdf_sel['color'])))

        m.add_tile_layer(
            url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            name="Google Satellite Hybrid",
            attribution="Google"
        )

        m.to_streamlit(height=650, width=700)


