import numpy as np
import streamlit as st
import pandas as pd
import joblib
Kmeans=joblib.load(".dist\kmeans_model.pkl")
scaler=joblib.load(".dist\scaler.pkl")
st.title("'Customer Segmentation'")
st.write("Enter Customer Details To know The Segmentation")
age=st.number_input("Enter Age:",min_value=5,max_value=130,value=18)
income=st.number_input("Enter the Income of Customer:",min_value=0,max_value=1500098,value=80000)
#features=["Age","Income",'NumWebPurchases','NumStorePurchases','NumWebVisitsMonth']
spent=st.number_input("Enter Total Money Spend in Store:",min_value=0,max_value=9999999,value=1000)
purchase=st.number_input("Enter The Products Bought in Online:",min_value=0,max_value=31,value=7)
purchase1=st.number_input("Enter The Products Bought in Store:",min_value=0,max_value=31,value=7)
visit=st.number_input("Enter Number of Visists per Month:",min_value=0,max_value=31,value=7)
inputdata=pd.DataFrame({
    "Age":[age],
    "Income":[income],
    "NumWebPurchases":[purchase],
    "NumStorePurchases":[purchase1],
    "NumWebVisitsMonth":[visit],
    "Total_spend":[spent]
})
input_scaled=scaler.transform(inputdata)
if st.button("Predict"):
    cluster=Kmeans.predict(input_scaled)
    st.success(f"predicted scale Cluster:{cluster}")