import pandas as pd
import numpy as np 
import plotly.express as px 
import math 
import plotly.graph_objs as go
import streamlit as st
import networkx as nx

st.header("Trial")

data=pd.read_csv("data_06_19.csv")

option_st=st.selectbox("Select focus area for curved network map", ("FA1", "FA2", "FA3", "FA4", "FA5", "FA6", "FA7", "FA8", "YRO"))



FA=data[data[option_st]==1]

id_IReNA=list(FA["ID"])
lat=list(FA["Latitude"])
long=list(FA["Longitude"])
net=list(FA["Network"])

graph=nx.Graph()

dict_by_id={}

for i in range(len(id_IReNA)):
    id_tempo=id_IReNA[i]
    tempo=[]

    if dict_by_id.get(id_tempo,0)!=0:
        continue
    else:
        tempo.append(net[i])
        tempo.append(lat[i])
        tempo.append(long[i])

        dict_by_id[id_tempo]=tempo


for i in id_IReNA:
    net=dict_by_id[i][0]
    lat=dict_by_id[i][1]
    long=dict_by_id[i][2]

    graph.add_node(i,Network=net,Latitude=lat, Longitude=long)


nodes_list=list(graph.nodes())

for i in range(0,len(nodes_list),1):
    for j in range(i+1,len(nodes_list),1):
        if graph.nodes[nodes_list[i]]["Network"] != graph.nodes[nodes_list[j]]["Network"]:
            graph.add_edge(nodes_list[i], nodes_list[j])
        else:
            continue


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
}


if st.button("generate graph!"):
    def circular_arc(start, end, num_points=20):
        """
        Calculate circular arc points between start and end.
        start, end: start and end points of the arc
        num_points: number of points to generate on the arc
        """
        start = np.array(start)
        end = np.array(end)
        
        # Calculate the midpoint and perpendicular vector for control point
        mid = (start + end) / 2
        
        perp = np.array([-(end[1] - start[1]), end[0] - start[0]])
        control = mid + 0.15 * perp  # Adjust 0.1 to control curvature
        
        # Generate points along the arc
        t = np.linspace(0, 1, num_points)
        arc = (1 - t)[:, np.newaxis] * (1 - t)[:, np.newaxis] * start + \
              2 * (1 - t)[:, np.newaxis] * t[:, np.newaxis] * control + \
              t[:, np.newaxis] * t[:, np.newaxis] * end
        
        return arc

    #Extract data to define the `scattermapbox` trace for graph nodes:
    node_data=list(graph.nodes(data=True))
    
    ids = [nl[0] for nl in node_data]
    lats = [nl[1]['Latitude'] for  nl in node_data]
    lons = [nl[1]['Longitude'] for  nl in node_data]
    Net= [nl[1]['Network'] for nl in node_data]
    
    sd=0.01
    
    lats = [sum(x) for x in zip(lats, list(np.random.normal(0, sd, len(lats))))]
    lons = [sum(x) for x in zip(lons, list(np.random.normal(0, sd, len(lons))))]
    
    tooltips = ids
    
    # Define the dict for the graph layout (i.e. node geographic locations)
    pos = {id: [la, lo] for id, la, lo in zip(ids, lats, lons)}
    
    # Get the map center
    x_center = (min(lons) + max(lons)) / 2
    y_center = (min(lats) + max(lats)) / 2
    
    fig_try = go.Figure(layout=go.Layout(width=1024, height=768))
    
    for network in network_color_map:
        net_ids = [ids[i] for i in range(len(Net)) if Net[i] == network]
        net_lats = [lats[i] for i in range(len(Net)) if Net[i] == network]
        net_lons = [lons[i] for i in range(len(Net)) if Net[i] == network]
        net_color = network_color_map[network]
    
        fig_tr.add_trace(go.Scattermapbox(
            lat=net_lats,
            lon=net_lons,
            mode='markers',
            text=net_ids,
            marker=dict(size=8, color=net_color),
            name=network,
            showlegend=True,
            hoverinfo='text'
        ))
    
    
    edge_list = list(graph.edges(data=True))
    pl_edges = [(item[0], item[1]) for item in edge_list]
    
    # Define the coordinates of the edge ends as 'MultiLineString' data type
    coords = []
    for e in pl_edges:
        start = pos[e[0]]
        end = pos[e[1]]
        arc = circular_arc(start, end)
        coords.append(arc)
    
    # Flatten the coordinates for plotly
    lines = []
    for arc in coords:
        for i in range(len(arc) - 1):
            lines.append(arc[i])
            lines.append(arc[i + 1])
            lines.append([None, None])  # to break the line
    
    line_lats, line_lons = zip(*lines)
    
    fig_tr.add_trace(go.Scattermapbox(
        mode="lines",
        lon=line_lons,
        lat=line_lats,
        line=dict(width=1, color='red'),
        opacity=0.02,
        hoverinfo='skip',
        showlegend=False
    ))
    
    
    fig_tr.update_layout(
        title_text="IReNA Members in "+FA_info+" Connected Across Network",
        title_x=0.5,
        font=dict(family='Balto', color='black'),
        autosize=False,
        width=1200,
        height=1200,
        hovermode='closest',
        mapbox=dict(  # accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(lat=y_center,
                        lon=x_center + 0.01),
            pitch=0,
            zoom=0.3,
            style='open-street-map'
        ),
        margin=dict(t=150)
    )
    st.plotly_chart(fig_tr)
      

