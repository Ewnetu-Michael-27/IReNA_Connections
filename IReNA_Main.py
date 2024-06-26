import streamlit as st 
import pandas as pd 
import numpy as np 
import altair as alt 
import pydeck as pdk 
from PIL import Image
import math
import pickle

st.set_page_config(
    page_title="IReNA Demo", 
    layout="wide",
    
)

st.sidebar.success("Select further options")

st.title("IReNA Demo")
image=Image.open("IReNA.png")
st.image(image)

st.header("The purpose of this page is to allow the user interact with IReNA data and learn about the members.", divider="red")

data=pd.read_csv("data_06_19.csv")


st.write("**Search members by**")

options_1=st.multiselect(
    "What parameter you want to search members by?, or just click apply to see the full list.",
    ["Name", "Institution", "Country", "Network", "Position", "Specific Focus Area Membership"], 
    )


col1, col2=st.columns([1,1])
with col1:
    if "Name" in options_1:
        First=st.text_input("First Name","")
        #First=First.lower()
        Last=st.text_input("Last Name","")
        #Last=Last.lower()
    if "Institution" in options_1:
        HI=data["Home Institution"].unique()
        options_2=st.multiselect("Choose Home Institution from the following list",
                             HI)
    if "Country" in options_1:
        Co=data["Country"].unique()
        options_3=st.multiselect("Choose Country from the following list", Co)
    
    if "Network" in options_1:
        Ne=data["Network"].unique()
        options_4=st.multiselect("Choose Network from the following list", Ne)

    if "Position" in options_1:
        Po=data["Position"].unique()
        option_5=st.multiselect("Choose Position of memeber from the following list", Po)

    if "Specific Focus Area Membership" in options_1:
        FA=["YRO","FA1","FA2","FA3","FA4","FA5","FA6","FA7","FA8"]
        option_6=st.multiselect("Choose Focus Area from the following options", FA)

    

with col2:
    result=data.copy()
    
    if "Name" in options_1:
        if len(First)>0:
            if len(Last)>0:
                result=result[(result["First"].str.contains(First, case=False, na=False)) & (result["Last"].str.contains(Last, case=False, na=False))].reset_index(drop=True)
            else:
                result=result[result["First"].str.contains(First, case=False, na=False)].reset_index(drop=True)
        elif len(Last)>0:
            result=result[result["Last"].str.contains(Last, case=False, na=False)].reset_index(drop=True)

    if "Institution" in options_1:
        result=result[result["Home Institution"].isin(options_2)].reset_index(drop=True)
    
    if "Country" in options_1:
        result=result[result["Country"].isin(options_3)].reset_index(drop=True)
    
    if "Network" in options_1:
        result=result[result["Network"].isin(options_4)].reset_index(drop=True)
    
    if "Position" in options_1:
        result=result[result["Position"].isin(option_5)].reset_index(drop=True)

    if "Specific Focus Area Membership" in options_1:
        result=result[(result[option_6]==1).all(axis=1)].reset_index(drop=True)

    if st.button("Apply"):
        st.dataframe(result)
    
st.write("")
st.markdown("***")
st.write("")

st.write("**From 2020 to 2024, IReNA has supported the publication of 149 papers.**")
st.write("Select year to view the papers")

with open("Paper_by_year.pkl", "rb") as f:
    dict_title_years=pickle.load(f)

col3, col4=st.columns([1,3])

with col3:
    options_7=st.selectbox("Select Year of Publication to view papers",
                             [2020,2021,2022,2023,2024]
                             )
with col4:
    st.write(str(len(dict_title_years[options_7]))+" Papers for the year: "+str(options_7))
    for i in dict_title_years[options_7]:
        st.write(i)

st.write("")
st.markdown("***")
st.write("")

st.write("You can also select a member to see the IReNA papers he/she has published.")

with open("apl_main.pkl", "rb") as f:
    apl_main=pickle.load(f)

ID=list(data["ID"])

col3b, col4b=st.columns([1,3])

with col3b:
    options_7b=st.selectbox("Select IReNA member to view his/her IReNA based publications",
                             ID
                             )
with col4b:
    if apl_main.get(options_7b,0)==0:
        st.write(options_7b, " has not published IReNA based paper")
    else:
        val=apl_main[options_7b]
        st.write(options_7b, " has published **", str(len(val)), "** IReNA based papers.")
        st.write("")
        for i in range(len(val)):
            st.write(f":red[{str(i+1)}]. ",val[i])
