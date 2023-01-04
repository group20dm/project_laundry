import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

with st.sidebar:
  selected = option_menu (
    menu_title = "Main menu",
    options = ["Data Analysis", "Feature Selection & SMOTE", "Model"],
    icons = ["bar-chart-line","box-arrow-down-right","robot"],
  )
 
if selected == "Data Analysis":
  st.title("Data Analysis")
  st.map()
if selected == "Feature Selection & SMOTE":
  st.title("Feature Selection & SMOTE")
if selected == "Model":
  st.title("Model")
#first page data analysis (skip)
#left side google map

#second page feature selection & smote
#left sub header boruta right is rfe

#third page model
