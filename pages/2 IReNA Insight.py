import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import plotly.graph_objs as go
import networkx as nx 
import plotly.express as px
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="IReNA Membership Insight", 
    layout="wide"
)

st.title("IReNA Membership Insight")

st.sidebar.success("Select further options")


st.write("**Publication Related Insight.**")

data_1a=pd.read_csv("HI_addy.csv")
data_pub=pd.read_csv("IReNA_pub_metrics.csv")
data_pub["Year"]=data_pub["bibcodes"].apply(lambda x: int(x[0:4]))


#**************************************************************************************
data_1=data_pub[["Total_Read", "Total_Download", "Total_Citation", "Year"]].groupby("Year").sum().reset_index()
data_2=data_pub[["Total_Read", "Total_Download", "Total_Citation", "Year"]].groupby("Year").count().reset_index()

data_2["Number of Journals"]=data_2["Total_Citation"]

fig_n = px.line(data_2, x="Year", y="Number of Journals", title='Increasing IReNA Journals Through The Years')
fig_n.update_layout(xaxis = dict(
        title="Year",
        tickmode = 'linear',
        tick0 = 2019,
        dtick = 1,
        range=[2019, 2025]
    ), 
    yaxis = dict(
        title="Number of Journals",
        tickmode = 'linear',
        tick0 = 0,
        dtick = 5
    )
    )
fig_n.add_annotation(
    x=2024, 
    y=data_2["Number of Journals"].iloc[-1],
    text="2024 data is incomplete",
    showarrow=True, 
    arrowhead=2,
    yshift=10
)

fig_n.update_traces(line_color='red')
fig_n.update_layout(hovermode="x unified")
st.plotly_chart(fig_n)

st.write("**IReNA papers have seen siginificant engagement from 2020-2024**")


fig_1M = go.Figure(
    data=[
        go.Bar(
            x=["Total Read", "Total Download", "Total Citation"],
            y=[sum(data_pub["Total_Read"]),sum(data_pub["Total_Download"]),sum(data_pub["Total_Citation"])],
            marker_color=["blue", "blue", "blue"]
        )
    ],
    layout=go.Layout(
        title="Total Read, Download, and Citation of IReNA Papers",
        xaxis=dict(
            title="Metric"
        ),
        yaxis=dict(
            title="Value"
        )
    )
)

fig_1M.update_traces(width=0.2)

st.plotly_chart(fig_1M)

col1a, col2a=st.columns([1,1])

with col1a:
    fig_n2= px.bar(data_1[["Total_Read", "Total_Download", "Year"]].melt(id_vars='Year', var_name='Category', value_name='Value')
                 , x='Year', y='Value', 
                 color='Category', barmode='group', title="Downloading and Reading Trends on IReNA Papers")
    st.plotly_chart(fig_n2)

with col2a:
    fig_n3 = px.bar(data_1[["Total_Citation","Year"]], x='Year', y='Total_Citation', title="Citation Trend on IReNA Papers")
    st.plotly_chart(fig_n3)

#**************************************************************************************************************************#############################


st.write("**Member Insitutions**")
#data_1a is defined earlier from HD_addy
st.write(str(len(data_1a["Institution"])), "Member Institutions in ",str(len(data_1a["Country"].unique())),"Countries. See the map below to see their location.")

#Get the map center:
ids=list(data_1a["Institution"])
lats=list(data_1a["Lat"])
lons=list(data_1a["Long"])

x_center = (min(lons) + max(lons))/2
y_center = (min(lats) + max(lats))/2

mapbox_access_token = 'pk.eyJ1IjoiZXduZXR1bWkiLCJhIjoiY2x4bWJpa21wMDI1cjJrcHZ6Y3J5NXowZCJ9.Uh0I7zo7txKT6h4MuVy-fQ'

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
            mapbox=dict(accesstoken=mapbox_access_token,
                        bearing=0,
                        center=dict(lat=y_center,
                                    lon=x_center+0.01),
                        pitch=0,
                        zoom=1, 
                        style='mapbox://styles/ewnetumi/clxp33i6h031o01qk1yklbqq1'
                        ),
            margin=dict(t=150)
            )

st.plotly_chart(fig)



st.write("See memeber instituions below by selecting countiries")
col9, col10=st.columns([1,3])

with col9:
    Cou=list(data_1a["Country"].unique())
    options_10=st.selectbox("Choose Country", Cou)

with col10:
    if st.button(" Apply "):
        d=list(data_1a[data_1a["Country"]==options_10]["Institution"])
        st.write(pd.DataFrame(d, columns=["List of Institutions"]))

st.write("")
st.markdown("***")
st.write("")

st.header("Focus Area and Project Membership/ Event Participation")
st.write("")
st.markdown(""" 
IReNA is organized into eight focus areas (FA) that address 
            problem at the forefront of science. In all eight areas there is a particularly strong benefit of connecting the 
            complementary expertise and capabilities of the various international networks and coordinate research internationally. """)

st.write("")
st.write("YRO: Young Researchers Organization")
st.write("FA1: Nuclear Reaction Rates")
st.write("FA2: Stellar Abundances")
st.write("FA3: Dense Matter in Supernovae and Neutron Star Mergers")
st.write("FA4: r-process Experiments")
st.write("FA5: Computer Models")
st.write("FA6: Nuclear Data for Astrophysics")
st.write("FA7: Weak Interactions")
st.write("FA8: Professional Development and Broadening Participation")
st.write("")

#############################################################################################################
#data_2=pd.read_csv("data_main_09_rep.csv")
def init_connection():
    return create_engine("postgresql://IReNA_membership_owner:LiheZVPl1DS0@ep-still-frost-a8bel6ne.eastus2.azure.neon.tech/IReNA_membership?sslmode=require")

@st.cache_resource
def get_engine():
    return init_connection()

engine=get_engine()
with engine.connect() as conn:
    query=text("SELECT * FROM members")
    data_2=pd.read_sql(query, conn)
###############################################################################################################

FA=["YRO","FA1","FA2","FA3","FA4","FA5","FA6","FA7","FA8"]
option_10A=st.selectbox("Choose Focus Area from the following options to see membership distribution", FA)

result=data_2[data_2[option_10A]==1]


ids=list(result["ID"])
lats=list(result["Latitude"])
lons=list(result["Longitude"])
Net=list(result["Network"])


graph=nx.Graph()

dict_by_id={}

for i in range(len(ids)):
    id_tempo=ids[i]
    tempo=[]

    if dict_by_id.get(id_tempo,0)!=0:
        continue
    else:
        tempo.append(Net[i])
        tempo.append(lats[i])
        tempo.append(lons[i])
        
        dict_by_id[id_tempo]=tempo

for i in ids:
    net=dict_by_id[i][0]
    graph.add_node(i,Network=net)

nodes_list=list(graph.nodes())

for i in range(0,len(nodes_list),1):
    for j in range(i+1,len(nodes_list),1):
        if graph.nodes[nodes_list[i]]["Network"] != graph.nodes[nodes_list[j]]["Network"]:
            graph.add_edge(nodes_list[i], nodes_list[j])
        else:
            continue



sd=0.01

lats=[sum(x) for x in zip(lats, list(np.random.normal(0, sd, len(lats))))]
lons=[sum(x) for x in zip(lons, list(np.random.normal(0, sd, len(lons))))]
     
#dpcapacity = [nl[1]['dpcapacity'] for  nl in node_data]
tooltips = ids

#Define the dict for the graph layout (i.e. node geographic locations):
pos = {id: [lo, la]  for id, lo, la in zip(ids, lons, lats)}

#Get the map center:
x_center = (min(lons) + max(lons))/2
y_center = (min(lats) + max(lats))/2

edge_list = list(graph.edges(data=True))

pl_edges = [(item[0], item[1]) for item in edge_list]

coords=[]
for e in pl_edges:
    coords.append([ [pos[e[0]][0], pos[e[0]][1]],  [pos[e[1]][0], pos[e[1]][1]] ])  

#Assign color
network_color_map={
    'BRIDGCE':'#7f7f7f', #gray
    'CaNPAN':'#98df8a', #light green
    'ChETEC':'#e377c2', #pink
    'ChETEC Infra':'#ff7f0e', #Orange
    'CRC881':'#ffbb78',#light Orange 
    'EMMI':'#2ca02c', #green
    'IANNA':'yellow', #yellow
    'JINA-CEE/CeNAM':'#1f77b4', #blue
    'NuGRID':'#17becf', #cyan
    'UKAKUREN':'#bcbd22', #yellow-green
}

node_colors=[network_color_map[net] for net in Net]

fig_1= go.Figure()

for network in network_color_map:
    net_ids = [ids[i] for i in range(len(Net)) if Net[i] == network]
    net_lats = [lats[i] for i in range(len(Net)) if Net[i] == network]
    net_lons = [lons[i] for i in range(len(Net)) if Net[i] == network]
    net_color = network_color_map[network]
    
    fig_1.add_trace(go.Scattermapbox(
        lat=net_lats,
        lon=net_lons,
        mode='markers',
        text=net_ids,
        marker=dict(size=8, color=net_color),
        name=network,
        showlegend=True,
        hoverinfo='text'
    ))

width_val={
        "FA1":0.6,
        "FA2":0.85,
        "FA3":1,
        "FA4":1,
        "FA5":0.7,
        "FA6":0.8,
        "FA7":1,
        "FA8":1,
        "YRO":1
    }
layers = [dict(sourcetype = 'geojson',
               source={"type": "Feature",
                       "geometry": {"type": "MultiLineString",
                                    "coordinates": coords}
                      },
             color= 'red',
             type = 'line',
              opacity=0.015,   
             line=dict(width=width_val[option_10A]),
            )]


#mapboxt = open(".mapbox_token").read().rstrip() #my mapbox_access_token  must be set only for special mapbox style
fig_1.update_layout(title_text="IReNA Members in "+option_10A+" Connected Across Network", title_x=0.5,
              font=dict(family='Balto', color='black'),
              autosize=False,
              width=1500,
              height=1000,
              hovermode='closest',
    
              mapbox=dict(accesstoken=mapbox_access_token,
                          layers=layers,
                          bearing=0,
                          center=dict(lat=y_center,
                                      lon=x_center+0.01),
                          pitch=0,
                          zoom=1.4,
                          style='open-street-map'
                         ),
            margin=dict(t=150)
            )

if st.button("Click to see chart"):
    st.write(f"**{len(ids)}** members in **{option_10A}**")
    st.plotly_chart(fig_1)

    st.write("")
    st.header("Participant Information about "+ option_10A)

    col1b, col2b, col3b=st.columns([1,1,1])

    with col1b:
        ti="Distribution of Position within "+ option_10A
        fig_1b=px.pie(result[["Position", "First"]].groupby("Position").count().reset_index().rename(columns={"First":"Count"}),
            values="Count", names="Position", title=ti)
        fig_1b.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_1b)

    with col2b:
        ti_2="Top 10 Countries of Participants within "+ option_10A
        fig_2b=px.bar(result[["Country", "First"]].groupby("Country").count().sort_values(by=["First"], ascending=False).reset_index().head(10).rename(columns={"First":"Count"}),
            x="Country",
            y="Count"
        )
        fig_2b.update_layout(
            title=ti_2,
        )
        st.plotly_chart(fig_2b)
    with col3b:
        ti_3="Network Distribution of Participants within "+ option_10A
        fig_3b=px.bar(
            result[["Network", "First"]].groupby("Network").count().sort_values(by=["First"], ascending=False).reset_index().rename(columns={"First":"Count"}),
            x="Network",
            y="Count"
        )
        fig_3b.update_layout(
            title=ti_3,
        )
        st.plotly_chart(fig_3b)



