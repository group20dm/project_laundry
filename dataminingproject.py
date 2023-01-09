import streamlit as st
import pandas as pd
import seaborn as sns
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
  col1, col2 = st.columns(2)
  with col1:
    st.header("Google map")
    map_heatmap = folium.Map(location=[3.060525411,101.6105832], zoom_start=11,height = 50, width = 50)
    
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
    st.header("Total Number of Customers in each Days")
    days_count = data.groupby("date").size().reset_index()

    days_count.columns = ["date", "total_cust"]

    total_cust_fig = px.line(days_count, x = "date", y = "total_cust")
    total_cust_fig.update_layout(title = "Total Number of Customers in Each Days", xaxis_title = "Date", yaxis_title = "Total Number of Customers")
    st.write(total_cust_fig)
  
  with col1:
    st.header("Percentage of Sales in Each Month and Year")
    
    sales = data.copy()
    sales["month_year"] = sales.date.dt.strftime('%Y-%m')
    sales = sales.groupby("month_year").totalspent_rm.sum().reset_index()
    sales.columns = ["month_year", "sales"]

    perc_sale_fig  = make_subplots(1, 2, specs = [[{"type": "domain"}, {"type": "xy"}]], 
                        subplot_titles = ["Percentage of Sales<br>in Each Year and Month", "Total Sales in Each Year and Month"])
    perc_sale_fig.add_trace(go.Pie(hole= .5, labels= sales.month_year, values= sales.sales, legendgroup = "1"), row = 1, col = 1)
    perc_sale_fig.add_trace(go.Scatter(x = sales.month_year, y = sales.sales, showlegend = False, legendgroup = "2"), row = 1, col = 2)

    perc_sale_fig.update_layout(margin = {"l": 0, "r": 0, "b": 0, "t": 50}, legend_title_text = "Year and Month", 
                     yaxis1_title = "Year and Month", xaxis1_title = "Total Sales (RM)")
    st.write(perc_sale_fig)
  
  with col2:
    sns_heatmap = sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu")
    st.write(sns_heatmap)
    
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
