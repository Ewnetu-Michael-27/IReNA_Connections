import streamlit as st 
import pandas as pd 
import numpy as np 
import pydeck as pdk 
import pickle
import plotly.graph_objs as go


st.set_page_config(
    page_title="IReNA Membership Insight"
)

st.title("IReNA Membership Insight")

st.sidebar.success("Select further options")



data_1=pd.read_csv("HI_addy.csv")

st.write("**Starting with Insitutions**")
st.write(str(len(data_1["Institution"]))," Member Institutions. See the map below to see their location.")

#Get the map center:
ids=list(data_1["Institution"])
lats=list(data_1["Lat"])
lons=list(data_1["Long"])

x_center = (min(lons) + max(lons))/2
y_center = (min(lats) + max(lats))/2

fig= go.Figure(go.Scattermapbox(  #trace for nodes
            lat= lats,
            lon=lons,
            mode='markers',
            text=ids,
            marker=dict(size=8, 
                        color= "red", 
                        colorscale='matter'
                        ),
            showlegend=False,
            hoverinfo='text'
            ))            

fig.update_layout(title_text="Location of Member Institutions", title_x=0.5,
            font=dict(family='Balto', color='black'),
            autosize=False,
            width=1200,
            height=1200,
            hovermode='closest',
            mapbox=dict(#accesstoken=mapbox_access_token,
                        bearing=0,
                        center=dict(lat=y_center,
                                    lon=x_center+0.01),
                        pitch=0,
                        zoom=1, 
                        style='carto-positron'
                        ),
            margin=dict(t=150)
            )

st.plotly_chart(fig)



st.write("See memeber instituions below by selecting countiries")
col9, col10=st.columns([1,3])

with col9:
    Cou=list(data_1["Country"].unique())
    options_10=st.selectbox("Choose Country", Cou)

with col10:
    if st.button(" Apply "):
        d=list(data_1[data_1["Country"]==options_10]["Institution"])
        st.write(d)

st.write("")
st.markdown("***")
st.write("")

