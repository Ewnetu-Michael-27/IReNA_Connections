import streamlit as st 
import pandas as pd 
import numpy as np 
import networkx as nx
import pickle
import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F 
import networkx as nx
import plotly.graph_objs as go

st.set_page_config(
    page_title="IReNA Connections"
)

st.sidebar.success("Select further options")

st.title("IReNA Connections")

#loading the graph data
#graph_M=nx.read_gexf("graph_trial.gexf")

data=pd.read_csv("data_05_24_ID.csv")


with open("graph_trial.pkl", "rb") as f:
    graph_trial=pickle.load(f)


st.write("**In the 149 publications from 2020 to 2024, IReNA members were able to collaborate and form connections.**")
st.write("Select an IReNA member ID below to see their connections")


def get_connections(graph, node):
    try:
        connections=[(neighbor, graph[node][neighbor]["Connection"]) for neighbor in graph.neighbors(node)]

        Conn_sorted=sorted(connections, key=lambda x: x[1], reverse=True)
        
        return [neighbor for neighbor, _ in Conn_sorted]
    except nx.NetworkXError:
        return f"Node {node} does not exist in the graph"
    

col5, col6=st.columns([1,3])


node=list(graph_trial.nodes())
with open("pub_info.pkl", "rb") as f:
    dict_pub_info=pickle.load(f)


options_8=st.selectbox("Select an IReNA member ID", node)
nodes=get_connections(graph_trial, options_8)


if st.button("Apply Queery"):
    st.write(options_8, " is an author in ", str(dict_pub_info[options_8]), " out of the 149 papers. See connection below" )
    st.write(nodes)


    #Creating graph for selected node***********************************************************************
    graph_trial_2=nx.Graph()

    val=data[data["ID"]==options_8]
    Latitude=val["Latitude"].values[0]
    Longitude=val["Longitude"].values[0]

    graph_trial_2.add_node(options_8, Latitude=Latitude, Longitude=Longitude)

    for i in nodes:

        val=data[data["ID"]==i]
        Latitude=val["Latitude"].values[0]
        Longitude=val["Longitude"].values[0]

        graph_trial_2.add_node(i, Latitude=Latitude, Longitude=Longitude)

    for i in range(len(nodes)):
        graph_trial_2.add_edge(options_8, nodes[i])   

    #Extract data to define the `scattermapbox` trace for graph nodes:
    node_data=list(graph_trial_2.nodes(data=True))

    ids = [nl[0] for nl in node_data]
    lats = [nl[1]['Latitude'] for  nl in node_data]
    lons = [nl[1]['Longitude'] for  nl in node_data]

    sd=0.01

    lats=[sum(x) for x in zip(lats, list(np.random.normal(0, sd, len(lats))))]
    lons=[sum(x) for x in zip(lons, list(np.random.normal(0, sd, len(lons))))]
        
    #dpcapacity = [nl[1]['dpcapacity'] for  nl in node_data]
    #tooltips = [f"{nl[1]['name']}<br>dpcapacity: {nl[1]['dpcapacity']}" for nl in node_data]  

    #Define the dict for the graph layout (i.e. node geographic locations):
    pos = {id: [lo, la]  for id, lo, la in zip(ids, lons, lats)}

    #Get the map center:
    x_center = (min(lons) + max(lons))/2
    y_center = (min(lats) + max(lats))/2

    fig= go.Figure(go.Scattermapbox(  #trace for nodes
                lat= lats,
                lon=lons,
                mode='markers',
                text=ids,
                marker=dict(size=8, 
                            #color= dpcapacity, 
                            colorscale='matter'
                            ),
                showlegend=False,
                hoverinfo='text'
                ))            

    edge_list = list(graph_trial_2.edges(data=True))

    pl_edges = [(item[0], item[1]) for item in edge_list]


    coords=[]
    for e in pl_edges:
        coords.append([ [pos[e[0]][0], pos[e[0]][1]],  [pos[e[1]][0], pos[e[1]][1]] ])  


    layers = [dict(sourcetype = 'geojson',
                source={"type": "Feature",
                        "geometry": {"type": "MultiLineString",
                                        "coordinates": coords}
                        },
                color= 'red',
                type = 'line',
                opacity=0.5,   
                line=dict(width=1),
                )]

    fig.update_layout(title_text="Graph on map", title_x=0.5,
                font=dict(family='Balto', color='black'),
                autosize=False,
                width=1200,
                height=1200,
                hovermode='closest',
        
                mapbox=dict(#accesstoken=mapbox_access_token,
                            layers=layers,
                            bearing=0,
                            center=dict(lat=y_center,
                                        lon=x_center+0.01),
                            pitch=0,
                            zoom=0.3,
                            style='open-street-map'
                            ),
                margin=dict(t=150)
                )

    st.plotly_chart(fig)


st.write("")
st.markdown("***")
st.write("")
