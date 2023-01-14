import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE

from scipy.stats import chi2_contingency
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import time
from tqdm import tqdm
 
import json
import folium
import base64
import numpy as np
import xgboost as xgb
import geopandas as gpd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from shapely.geometry import shape

from streamlit.components.v1 import html
from streamlit_folium import st_folium,folium_static
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_option_menu import option_menu
from boruta import BorutaPy
from xgboost import XGBClassifier, XGBRegressor

import ast
from yellowbrick.cluster import silhouette_visualizer

from ml_tools import *

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall, AUC
from keras.utils import plot_model
from keras import backend as K


from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import pickle

st.set_page_config(layout="wide")
mpl.rcParams.update(mpl.rcParamsDefault)

#navbar
with st.sidebar:
    selected = option_menu (
    menu_title = "Main menu",
    options = ["Data Analysis", "Classification", "Regression", "Classification prediction" , "Regression prediction"],
    icons = ["bar-chart-line","diagram-3","graph-up","diagram-3","graph-up"],
  )
#   selected = st.radio("Main menu",["Data Analysis", "Classification", "Regression", "Classification prediction" , "Regression prediction"])
  
  
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
    sns_heatmap, ax = plt.subplots(1,1, figsize=(10,10))
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
    sns.countplot(x = "wash_item", data = data,ax=ax)
    st.write(sns_countplot)
  
  
  #Outliers
  st.header("Outliers")
  def display_outliers(data, title):
    outliers = data.select_dtypes([float, int])
    n_col = len(outliers.columns)

    fig, axes = plt.subplots(n_col // 2, 2, figsize = (15, 15))
    for ax, col in zip(axes.flatten(), outliers.columns):
      sns.boxplot(x = col, data = outliers, ax = ax)

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
  
  display_missing, ax = plt.subplots(1,1, figsize=(15, 2))
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
  compare_missing_outliers, axes = plt.subplots(1, 2, figsize = (15, 5))
  display_missing_counts(data, "Number of Missing Values in each Features \nBefore Missing Values Handling", axes[0])
  display_missing_counts(data_copy, "Number of Missing Values in each Features \nAfter Missing Values Handling", axes[1])

  compare_missing_outliers.tight_layout()
  st.write(compare_missing_outliers)
  outliers2 = display_outliers(data, "Box plot for each Numerical Features Before Missing Values Handling")
  st.write(outliers2)
  outliers3 = display_outliers(data_copy, "Box plot for each Numerical Features After Missing Values Handling")
  st.write(outliers3)
  
  data = data_copy.copy()
  data["year"] = data.date.dt.year
  data["month"] = data.date.dt.month
  data["day"] = data.date.dt.day

  data.time = data.time.str.replace(";", ":")

  data[['hour', 'minute', "second"]] = data.time.str.split(":", expand = True).astype(int)
  data = data.drop(columns = ["date", "time"])

  data = data[data.nunique(dropna = False)[data.nunique(dropna = False) > 1].index]
  data.drop(columns = ["address", "city_geometry"], inplace = True)
#   def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
  
#   export_as_pdf = st.button("Export Report")
  
#   if export_as_pdf:
#       pdf = FPDF()
#       pdf.add_page()
#       html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
#       st.markdown(html, unsafe_allow_html=True)
  
#second page
if selected == "Classification":
  st.title("Classification")
  st.header("Feature Selection")
  
  
  
  #columns
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Boruta")
    boruta = Image.open('top10_boruta.jpg')
    st.image(boruta, caption='Top 10 boruta')
    
  with col2:
    st.subheader("RFE") 
    RFE = Image.open('top10_RFE.jpg')
    st.image(RFE, caption='Top 10 RFE')
  
  st.header("SMOTE")
  SMOTE = Image.open('SMOTE.jpg')
  st.image(SMOTE, caption='SMOTE')
  
  st.header("Model Construction")
  st.subheader("Comparison between Classifiers")
  compare_class = Image.open('compare_class.jpg')
  st.image(compare_class, caption='Comparison between Classifiers')
  
  st.header("Model Improvement")
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Metrics score")
    model_improve = Image.open('model_improve.jpg')
    st.image(model_improve, caption='Metrics score')
    
  with col2:
    st.subheader("Confusion Metrics") 
    model_improve_conf = Image.open('model_improve_conf.jpg')
    st.image(model_improve_conf, caption='Confusion Metrics')
#third page
if selected == "Regression":
  st.title("Regression")
  st.header("Feature Selection")
  
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Boruta")
    reg_boruta = Image.open('reg_boruta.jpg')
    st.image(reg_boruta, caption='Top 10 boruta')
    
  with col2:
    st.subheader("RFE") 
    reg_rfe = Image.open('reg_rfe.jpg')
    st.image(reg_rfe, caption='Top 10 RFE')
    
  st.header("Model Improvement")
  reg_model_improve = Image.open('reg_model_improve.jpg')
  st.image(reg_model_improve, caption='Metrics score')
 
if selected == "Classification prediction":
  st.title("Classification prediction")
  
  data = pd.read_csv("encoded_data.csv")
  
  cols = list(data.columns)
  cols.remove("wash_item")

  user_input = {}
  pickle_files = os.listdir("pickle_files")

  def find_encoder(col):
      for file in pickle_files:
          if col in file: 
              return True

      return False 

  numerical_input = ["age_range", "timespent_minutes", "buydrinks", "totalspent_rm", "num_of_baskets", "year"]

  nearby_laundries = ["Super Dryclean Sdn Bhd", "Jag Nasmech Sdn Bhd", "Zack Laundry", "Dobi Auto Sdn Bhd", "Dobi Pro Enterprise", "Laundry Bar (Pantai Hillpark) - 24 Hours"]

  for col in cols:

        uniq = data[col].unique()

        found_encoder = False
        if find_encoder(col): 
            encoder = pickle.load(open(f"pickle_files/{col}.pkl", "rb"))
            uniq = encoder.inverse_transform(uniq)
            found_encoder = True

        if col in numerical_input: selected = st.number_input(col)
        elif col == "month": selected = st.number_input(col, min_value = 1, max_value = 12)
        elif col in ["hour", "minute", "second"]: selected = st.number_input(col, min_value = 0, max_value = 60)
        elif col in nearby_laundries: st.selectbox(col, ["yes", "no"])
        else: selected = st.selectbox(col, uniq)

        if found_encoder: 
            user_input[col] = [encoder.transform([selected])]
        elif col in nearby_laundries: 
            user_input[col] = [0] if selected == "no" else [1]
        else: 
            user_input[col] = [selected]

  user_input = pd.DataFrame(user_input)
  
  scaler = pickle.load(open("pickle_files/class_scaler.pkl", "rb"))
  scaled = pd.DataFrame(scaler.transform(user_input), columns = user_input.columns)

  selected_washer = data[data.wash_item != 2]
  X = selected_washer.drop(columns = "wash_item")
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  X = pd.DataFrame(X_scaled, columns = X.columns)
  y = selected_washer.wash_item.copy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42, stratify = y)
  early_stopping = EarlyStopping(patience=3)
  sm = SMOTE(random_state=42)
  X_train_res, y_train_res = sm.fit_resample(X_train,y_train)
  mlp_washer_improved = Sequential([
      Dense(400, activation = "relu", input_shape = (X_train_res.shape[1],)),
      Dense(350, activation = "relu"),
      Dense(200, activation = "relu"),
      Dense(150, activation = "relu"),
      Dense(100, activation = "relu"),
      Dense(1, activation = "sigmoid")
  ])

  mlp_washer_improved.compile(optimizer='adam', loss="binary_crossentropy", metrics = ['acc', Precision(), Recall(), AUC()])
  
  
  mlp_washer_history = mlp_washer_improved.fit(X_train_res, y_train_res, validation_split=0.2, epochs=50, callbacks=[early_stopping])
  predict = int(mlp_washer_improved.predict(scaled)[0] > .5)
  predict = "clothes" if predict == 1 else "blankets"
   
  st.write(f"Your predicted value is : {predict}.")
  
if selected == "Regression prediction":
  st.title("Regression prediction")
  
  data = pd.read_csv("encoded_data.csv")
  
  cols = list(data.columns)
  cols.remove("totalspent_rm")

  user_input = {}
  pickle_files = os.listdir("pickle_files")

  def find_encoder(col):
      for file in pickle_files:
          if col in file: 
              return True

      return False 

  numerical_input = ["age_range", "timespent_minutes", "buydrinks", "totalspent_rm", "num_of_baskets", "year"]

  nearby_laundries = ["Super Dryclean Sdn Bhd", "Jag Nasmech Sdn Bhd", "Zack Laundry", "Dobi Auto Sdn Bhd", "Dobi Pro Enterprise", "Laundry Bar (Pantai Hillpark) - 24 Hours"]

  for col in cols:

        uniq = data[col].unique()

        found_encoder = False
        if find_encoder(col): 
            encoder = pickle.load(open(f"pickle_files/{col}.pkl", "rb"))
            uniq = encoder.inverse_transform(uniq)
            found_encoder = True

        if col in numerical_input: selected = st.number_input(col)
        elif col == "month": selected = st.number_input(col, min_value = 1, max_value = 12)
        elif col in ["hour", "minute", "second"]: selected = st.number_input(col, min_value = 0, max_value = 60)
        elif col in nearby_laundries: st.selectbox(col, ["yes", "no"])
        else: selected = st.selectbox(col, uniq)

        if found_encoder: 
            user_input[col] = [encoder.transform([selected])]
        elif col in nearby_laundries: 
            user_input[col] = [0] if selected == "no" else [1]
        else: 
            user_input[col] = [selected]

  user_input = pd.DataFrame(user_input)
  
  scaler = pickle.load(open("pickle_files/reg_scaler.pkl", "rb"))
  scaled = pd.DataFrame(scaler.transform(user_input), columns = user_input.columns)
  scaled = scaled[['city_district', 'shop', 'office', 'building', 'man_made',
         'house_number', 'amenity', 'hamlet', 'suburb', 'neighbourhood']]
  st.write(scaled)

  model = pickle.load(open("pickle_files/best_reg.pkl", "rb"))
  predict = model.predict(scaled)[0]
  
  st.write(f"Your predicted value is : RM {predict:.2f}.")
  
with st.sidebar:  
if selected == "Data Analysis":
  if st.button("Download PDF"):
       html(
           f"""
               <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/1.5.3/jspdf.debug.js"></script>
               <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
               <script>{open("download.js").read()}

               </script>
               """,height = 0,width = 0)
