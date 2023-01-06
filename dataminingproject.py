import streamlit as st
import pandas as pd
import folium
import geopandas

from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

#navbar
with st.sidebar:
  selected = option_menu (
    menu_title = "Main menu",
    options = ["Data Analysis", "Feature Selection & SMOTE", "Model"],
    icons = ["bar-chart-line","box-arrow-down-right","robot"],
  )
#first page
if selected == "Data Analysis":
  st.title("Data Analysis")
  
  st.header("Dataset")
  df = pd.read_csv('dataset.csv')
  st.dataframe(df)
  
  #columns
  col1, col2 = st.columns(2)
  with col1:
    st.header("Google map")
    map_heatmap = folium.Map(location=[3.060525411,101.6105832], zoom_start=11)
    
    # Filter the DF for columns, then remove NaNs
    heat_df = df[["latitude", "longitude"]]
    heat_df = heat_df.dropna(axis=0, subset=["latitude", "longitude"])

    # List comprehension to make list of lists
    heat_data = [
        [row["latitude"], row["longitude"]] for index, row in heat_df.iterrows()
    ]

    # Plot it on the map
    HeatMap(heat_data).add_to(map_heatmap)
    st_folium(map_heatmap)
    
#     msia_geojson=geopandas.read_file('stanford-zd362bc5680-geojson.json')
#     msia_geojson.plot()
    
  with col2:
    st.header("2nd col")
    msia_geojson=geopandas.read_file('stanford-zd362bc5680-geojson.json')
    maps= folium.Choropleth(geo_data = msia_geojson,
                           data = df,
                           key_on='feature.properties.name',
                           fill_color='YlOrRd',
                           fill_opacity=0.7,
                           line_opacity=0.2,
                           legend_name=dicts[data],
                           highlight=True,
                           reset=True).add_to(map_sby)
    st_folium(map_sby)
    
#second page
if selected == "Feature Selection & SMOTE":
  st.title("Feature Selection & SMOTE")
  
  #columns
  col1, col2 = st.columns(2)
  with col1:
    st.header("Boruta")
    st.image("https://static.streamlit.io/examples/cat.jpg")
  with col2:
    st.header("RFE")
    st.image("https://static.streamlit.io/examples/cat.jpg")
    
#third page
if selected == "Model":
  st.title("Model")
#first page data analysis (skip)
#left side google map

#second page feature selection & smote
#left sub header boruta right is rfe

#third page model
