import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import plotly.graph_objs as go
import networkx as nx 


st.set_page_config(
    page_title="IReNA Membership Insight", 
    layout="wide"
)

st.title("IReNA Membership Insight")

st.sidebar.success("Select further options")



data_1=pd.read_csv("HI_addy.csv")

st.write("**Starting with Insitutions**")
st.write(str(len(data_1["Institution"])), "Member Institutions in ",str(len(data_1["Country"].unique())),"Countries. See the map below to see their location.")

#Get the map center:
ids=list(data_1["Institution"])
lats=list(data_1["Lat"])
lons=list(data_1["Long"])

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
    Cou=list(data_1["Country"].unique())
    options_10=st.selectbox("Choose Country", Cou)

with col10:
    if st.button(" Apply "):
        d=list(data_1[data_1["Country"]==options_10]["Institution"])
        st.write(d)

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

data_2=pd.read_csv("data_06_19.csv")


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
    'EMMI':'#2ca02c', #green
    'CeNAM':'#9467bd', #Purple 
    'UKAKUREN':'#bcbd22', #yellow-green
    'NuGRID':'#17becf', #cyan
    'ChETEC Infra':'#ff7f0e', #Orange
    'ChETEC':'#e377c2', #pink
    'BRIDGCE':'#7f7f7f', #gray
    'CaNPAN':'#98df8a', #light green
    'CRC881':'#ffbb78',#light Orange 
    'JINA-CEE':'#1f77b4', #blue
    'IANNA':'yellow', #yellow
    'ChETEC, NuGRID': '#8c564b' #brown
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

layers = [dict(sourcetype = 'geojson',
               source={"type": "Feature",
                       "geometry": {"type": "MultiLineString",
                                    "coordinates": coords}
                      },
             color= 'red',
             type = 'line',
              opacity=0.015,   
             line=dict(width=1),
            )]

#mapboxt = open(".mapbox_token").read().rstrip() #my mapbox_access_token  must be set only for special mapbox style
fig_1.update_layout(title_text="IReNA Members in "+option_10A+" Connected Across Network", title_x=0.5,
              font=dict(family='Balto', color='black'),
              autosize=False,
              width=1200,
              height=1200,
              hovermode='closest',
    
              mapbox=dict(accesstoken=mapbox_access_token,
                          layers=layers,
                          bearing=0,
                          center=dict(lat=y_center,
                                      lon=x_center+0.01),
                          pitch=0,
                          zoom=0.3,
                          style='mapbox://styles/ewnetumi/clxus5jj7049j01qj91i6223u'
                          #style='open-street-map'
                         ),
            margin=dict(t=150)
            )

if st.button("Click to see chart"):
    st.write(f"**{len(ids)}** members in **{option_10A}**")
    st.plotly_chart(fig_1)
