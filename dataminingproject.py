import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import folium
import base64
import numpy as np
import xgboost as xgb

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from shapely.geometry import shape

from streamlit_folium import st_folium,folium_static
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_option_menu import option_menu

from boruta import BorutaPy

from fpdf import FPDF
from tempfile import NamedTemporaryFile

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import normalize

from ml_tools import *

from imblearn.over_sampling import SMOTE

st.set_page_config(layout="wide")

#navbar
with st.sidebar:
  selected = option_menu (
    menu_title = "Main menu",
    options = ["Data Analysis", "Classification", "Regression"],
    icons = ["bar-chart-line","diagram-3","graph-up"],
  )
  
data = pd.read_csv('analytical_dataset.csv')

#first page
if selected == "Data Analysis":
  st.title("Data Analysis")
  #read csv
  st.header("Dataset")
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
  outliers1 = display_outliers(data, "Box plot for each Numerical Features Before Missing Values Handling")
  st.write(outliers1)
  
  
  #Missing Values Handling
  st.header("Missing Values Handling")
  def display_missing_counts(data, title, ax = None):
    ax = data.isna().sum().plot.bar(ax = ax)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Number of Missing Values")
    ax.set_title(title)

    return ax
  display_missing, ax = plt.subplots(figsize=(15, 2))
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
  compare_missing_outliers, axes = plt.subplots(1,2, figsize = (15, 5))
  display_missing_counts(data, "Number of Missing Values in each Features \nBefore Missing Values Handling", axes[0])
  display_missing_counts(data_copy, "Number of Missing Values in each Features \nAfter Missing Values Handling", axes[1])

  compare_missing_outliers.tight_layout()
  st.write(compare_missing_outliers)
  outliers2 = display_outliers(data, "Box plot for each Numerical Features Before Missing Values Handling")
  st.write(outliers2)
  outliers3 = display_outliers(data_copy, "Box plot for each Numerical Features After Missing Values Handling")
  st.write(outliers3)
  
  def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

  df = load_iris(as_frame=True)["data"]


  figs = []

  for col in df.columns:
      fig, ax = plt.subplots()
      ax.plot(df[col])
      st.pyplot(fig)
      figs.append(fig)

  export_as_pdf = st.button("Export Report")

  if export_as_pdf:
      pdf = FPDF()
      for fig in figs:
          pdf.add_page()
          with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                  fig.savefig(tmpfile.name)
                  pdf.image(tmpfile.name, 10, 10, 200, 100)
      html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
      st.markdown(html, unsafe_allow_html=True)
  
#second page
if selected == "Classification":
  st.title("Classification")
  st.header("Feature Selection")
  
  X = data.drop(columns = "washer_no")
  y = data.washer_no.copy()

  y[y != 3] = 0
  y[y == 3] = 1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42, stratify = y)
  
  #columns
  col1, col2 = st.columns(2)
  with col1:
    st.header("Baseline Model (without Feature Selection & SMOTE)")
    clf = xgb.XGBClassifier(random_state = 42, n_jobs = -1)

    models = {"XGBoost": xgb.XGBClassifier(random_state = 42, n_jobs = -1)}
    # model = clf
    # model.fit(X_train, y_train)
    # model.predict_proba(X_test)

    models = Model(models)
    metric = Metric(X_train, X_test, y_train, y_test, "binary")
    metric_score = metric.score(models = models)
    st.write(metric_score)
    
  with col2:
    st.header("RFE")
    conf_mat = metric.conf_mat(figsize = (8,3))
    st.write(conf_mat)
    
#third page
if selected == "Model":
  st.title("Model")
#first page data analysis (skip)
#left side google map

#second page feature selection & smote
#left sub header boruta right is rfe

#third page model
