## Libraries/Modules##
## Basic libraries
import pandas as pd

import numpy as np

 ## MOdel loading Libraries
import joblib

## UI & Logic libraries
import streamlit as st

## Loading trained model files
ohe = joblib.load("ohe.pkl")
sc = joblib.load("sc.pkl")
model = joblib.load("crop_mlr.pkl")

## UI & Logic code
st.header("Crop Yield Prediction Estimation")
st.write("The Yield is based on the crop for day")
st.image("pic1.jpg")

df = pd.read_csv("Crop prediction.csv")
st.write("Crop Yield Table")
st.dataframe(df.head(5))

st.subheader("Enter the crop prediction to the yield")

col1,col2,col3=st.columns(3)
with col1:
 Area = st.selectbox("Area:",df.Area.unique())
with col2:
 Item = st.selectbox("Item:",df.Item.unique())
with col3:
 Year = st.number_input("Year")

col4,col5,col6 = st.columns(3)
with col4:
 average_rain_fall_mm_per_year = st.number_input("average_rain_fall_mm_per_yea")
with col5:
 pesticides_tonnes = st.number_input("pesticides_tonnes")
with col6:
 avg_temp = st.number_input("avg_temp")

 ################# Logic Code ###############
 if st.button("Estimate"):
    row = pd.DataFrame([[Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]], columns=df.columns)
    

    st.write("Given Input Data:")
    
   # Onehot Encoding
    row_ohe = ohe.transform(row[['Area','Item']]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    
    row = pd.concat([row, row_ohe], axis=1)

    row=row.drop('Area',axis=1)
    row=row.drop('Item',axis=1)
    
    # Scaling
    row.iloc[:,0:4]= sc.transform(row.iloc[:,0:4])
    
    prediction = round(model.predict(row)[0],2)
    
    st.write(f"Estimated crop yield prediction: {prediction} ")
   