import streamlit as st
import pandas as pd
import folium
import geopandas
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from streamlit_folium import st_folium
from folium.plugins import HeatMap, FastMarkerCluster
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
  data = pd.read_csv('dataset.csv')
  st.dataframe(data)
  data.columns = data.columns.str.lower()
  categoricals = data.select_dtypes(include = object)

  for col in categoricals:
      data[col] = data[col].str.strip()

  data["date"] = pd.to_datetime(data["date"], infer_datetime_format = True)
  
  #columns
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.header("Google map")
    map_heatmap = folium.Map(location=[3.060525411,101.6105832], zoom_start=11)
    
    # Filter the DF for columns, then remove NaNs
    heat_data = data[["latitude", "longitude"]]
    heat_data = heat_data.dropna(axis=0, subset=["latitude", "longitude"])

    # List comprehension to make list of lists
    heat_list = [
        [row["latitude"], row["longitude"]] for index, row in heat_data.iterrows()
    ]

    # Plot it on the map
    HeatMap(heat_list).add_to(map_heatmap)
    FastMarkerCluster(heat_list).add_to(map_heatmap)
    st_folium(map_heatmap)
    
  with col2:
    st.header("2nd col")
    days_count = data.groupby("date").size().reset_index()

    days_count.columns = ["date", "total_cust"]

    fig = px.line(days_count, x = "date", y = "total_cust")
    fig.update_layout(title = "Total Number of Customers in Each Days", xaxis_title = "Date", yaxis_title = "Total Number of Customers")
    st.write(fig)
    
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
