import streamlit as st
import pandas as pd
import folium

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
    st.map(df)
    lat = df.latitude.tolist()
    lng = df.longitude.tolist()
    heat_df=df[["lat","lng"]]
    heat_data=list(zip(df.lat, df.lng))
    folium.plugins.HeatMap(heat_data).add_to(m)
    m
  with col2:
    st.header("2nd col")
    st.image("https://static.streamlit.io/examples/cat.jpg")
    
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
