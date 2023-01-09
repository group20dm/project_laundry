import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import folium

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

from shapely.geometry import shape
from plotly.subplots import make_subplots
from streamlit_folium import st_folium,folium_static
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

#navbar
with st.sidebar:
  selected = option_menu (
    menu_title = "Main menu",
    options = ["Data Analysis", "Classification", "Regression"],
    icons = ["bar-chart-line","diagram-3","graph-up"],
  )
  
  
#first page
if selected == "Data Analysis":
  st.title("Data Analysis")
  #read csv
  st.header("Dataset")
  data = pd.read_csv('analytical_dataset.csv')
  st.dataframe(data)
  data.columns = data.columns.str.lower()
  categoricals = data.select_dtypes(include = object)

  for col in categoricals:
      data[col] = data[col].str.strip()

  data["date"] = pd.to_datetime(data["date"], infer_datetime_format = True)
  
  
  #Total Sales in Each Area
  st.header("Total Sales in Each Area")
  analysis = data[["city", "city_geometry", "totalspent_rm"]].groupby(["city", "city_geometry"]).sum().reset_index()
  analysis.city_geometry = analysis.city_geometry.apply(lambda x: shape(json.loads(x)["geometries"][0]))
  analysis = gpd.GeoDataFrame(analysis, geometry = analysis.city_geometry, crs = "EPSG:4326")
  analysis.drop(columns = ["city_geometry"], inplace = True)

  m = folium.Map(location=[data.latitude.min(), data.longitude.max()], zoom_start = 10)

  folium.Choropleth(
      geo_data=analysis,
      data=analysis,
      columns=['city',"totalspent_rm"],
      key_on="feature.properties.city",
      fill_color='YlOrRd',
      fill_opacity=.9,
      line_opacity=0.2,
      highlight= True,
      line_color = "white",
      name = "Wills",
      show=False,
      nan_fill_color = "White"
  ).add_to(m)

  folium.features.GeoJson(data=analysis,
                          smooth_factor = 2,
                          style_function=lambda x: {'color':'black','fillColor':'transparent'},
                          tooltip=folium.features.GeoJsonTooltip(
                              fields=["city","totalspent_rm"],
                              aliases=["Area","Total Sales (RM)"], 
                              localize=True,
                              sticky=False,
                              style="""
                                  border: 2px solid black;
                                  border-radius: 5px;
                              """,

                              max_width=800,),

                                  highlight_function=lambda x: {'weight':4,'fillColor':'grey'},

                              ).add_to(m)
  folium_static(m,width = 1450)
  
  
  #columns
  col1, col2 = st.columns([1.3,2])
  with col1:
    
    
    #Total Number of Customers in each Days
    st.header("Total Number of Customers in each Days")
    days_count = data.groupby("date").size().reset_index()

    days_count.columns = ["date", "total_cust"]

    total_cust_fig = px.line(days_count, x = "date", y = "total_cust")
    total_cust_fig.update_layout(title = "Total Number of Customers in Each Days", xaxis_title = "Date", yaxis_title = "Total Number of Customers")
    st.plotly_chart(total_cust_fig,use_container_width = True)
    
  with col2:
    
    
    #Percentage of Sales in Each Month and Year
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
    st.plotly_chart(perc_sale_fig,use_container_width = True)
    
    
  col1, col2, col3 = st.columns([2,1.5,2])
  with col1:
    
    
    #Relationships between Variables
    st.header("Relationships between Variables")
    sns_heatmap, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu", ax=ax)
    st.write(sns_heatmap) 
  
  with col2:
    
    
    #What types of customers will likely to choose Washer No. 3 and Dryer No. 10?
    st.header("What types of customers will likely to choose Washer No. 3 and Dryer No. 10?")
    customers = data[(data.washer_no == 3) & (data.dryer_no == 10)]
    customers = customers.dropna(axis = 1)

    categ_type = pd.crosstab(customers.kids_category, customers.pants_type)
    st.write(categ_type)
    st.write("Most of customers who wear long pants will likely to choose Washer No.3 and Dryer No.10. From those customers, most of customers who are having toddler will likely to choose Washer No.3 and Dryer No.10.")
  
  with col3:
    
    
    #Do we need to perform data imbalance treatment?
    st.header("Do we need to perform data imbalance treatment?")
    sns_countplot, ax = plt.subplots(figsize=(8, 8))
    sns.countplot(x = "washer_no", data = data,ax=ax)
    st.write(sns_countplot)
  
  
  #Outliers
  st.header("Outliers")
  def display_outliers(data, title):
    outliers = data.select_dtypes([float, int])
    n_col = len(outliers.columns)

    fig, axes = plt.subplots(1, n_col, figsize = (15, 5))
    for idx, col in enumerate(outliers.columns):
        axes[idx] = sns.boxplot(y = col, data = outliers, ax = axes[idx])

    fig.suptitle(title)
    fig.tight_layout()

    return fig
  outliers = display_outliers(data, "Box plot for each Numerical Features Before Missing Values Handling")
  st.write(outliers)
  
  
  #Missing Values Handling
  st.header("Missing Values Handling")
  def display_missing_counts(data, title, ax = None):
    ax = data.isna().sum().plot.bar(ax = ax)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Number of Missing Values")
    ax.set_title(title)

    return ax
  display_missing, ax = plt.subplots(figsize=(10, 3))
  display_missing_counts(data, "Number of Missing Values in each Features \nBefore Missing Values Handling", ax = ax)
  st.write(display_missing)
  #
  categoricals = data.select_dtypes(object)
  categoricals.drop(columns = ["time"], inplace = True)

  data_copy = data.copy()
  for col in categoricals:
      data_copy[col] = data_copy[col].fillna("unknown")
  #
  other_missing_values = data_copy.isna().sum()
  other_missing_values = other_missing_values[other_missing_values > 0].index

  for col in other_missing_values:
      data_copy[col] = data_copy[col].fillna(data[col].median())
  #
  compare_missing_outliers, axes = plt.subplots(1,2, figsize = (10, 5))
  display_missing_counts(data, "Number of Missing Values in each Features \nBefore Missing Values Handling", axes[0])
  display_missing_counts(data_copy, "Number of Missing Values in each Features \nAfter Missing Values Handling", axes[1])

  compare_missing_outliers.tight_layout()
  display_outliers(data, "Box plot for each Numerical Features Before Missing Values Handling")
  display_outliers(data_copy, "Box plot for each Numerical Features After Missing Values Handling")
  st.write(compare_missing_outliers)
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
