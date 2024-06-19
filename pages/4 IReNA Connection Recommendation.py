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
    page_title="IReNA Connection Recommendation"
)

st.sidebar.success("Select further options")

st.title("IReNA Connection Recommendation")


data=pd.read_csv("data_06_19.csv")

with open("graph_trial.pkl", "rb") as f:
    graph_trial=pickle.load(f)

#Needed for indexing realted process in the link recommendation
dict_node_index_1={}
count=0
for node in graph_trial.nodes:
    dict_node_index_1[count]=node
    count+=1

dict_node_index_2={}
count=0
for node in graph_trial.nodes:
    dict_node_index_2[node]=count
    count+=1


#***********************************************************************
#Model
class HybridLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, fc_hidden_channels, alpha=0.5):
        super(HybridLinkPredictor, self).__init__()

        #gcn layers
        self.conv1=GCNConv(in_channels, hidden_channels)
        self.conv2=GCNConv(hidden_channels, out_channels)

        #fully connected layers
        self.fc1=torch.nn.Linear(in_channels, fc_hidden_channels)
        self.fc2=torch.nn.Linear(fc_hidden_channels, out_channels)

        self.alpha=alpha

    def encode(self,x,edge_index):
        #gcn part 
        x_gcn=self.conv1(x,edge_index)
        x_gcn=F.relu(x_gcn)
        x_gcn=self.conv2(x_gcn,edge_index)

        #fc part 
        x_fc=F.relu(self.fc1(x))
        x_fc=self.fc2(x_fc)

        #Combine gcn and FC outputs
        x_combined=(x_gcn*(1-self.alpha))+(x_fc*(self.alpha))
        return x_combined
        #return x_fc

    def decode(self, z,edge_label_index):
        return (z[edge_label_index[0]]*z[edge_label_index[1]]).sum(dim=1)
    
    def forward(self,x,edge_index,edge_label_index):
        z=self.encode(x,edge_index)
        return self.decode(z, edge_label_index)

with open("train_data.pkl", "rb") as f:
    train_data=pickle.load(f)

with open("data_main.pkl", "rb") as f:
    data_main=pickle.load(f)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(data):
    model.eval()

    with torch.no_grad():
        z=model.encode(data.x, train_data.edge_index)
        pos_pred=model.decode(z, data.edge_index).sigmoid()
        neg_pred=model.decode(z, data.neg_edge_index).sigmoid()
    
    pos_true=torch.ones(pos_pred.size(0))
    neg_true=torch.zeros(neg_pred.size(0))

    y_true=torch.cat([pos_true, neg_true], dim=0).cpu()
    y_pred=torch.cat([pos_pred, neg_pred], dim=0).cpu()

    return roc_auc_score(y_true, y_pred)


with open("model.pkl", "rb") as f:
    model=pickle.load(f)


def get_connections(graph, node):
    try:
        connections=[(neighbor, graph[node][neighbor]["Connection"]) for neighbor in graph.neighbors(node)]

        Conn_sorted=sorted(connections, key=lambda x: x[1], reverse=True)
        
        return [neighbor for neighbor, _ in Conn_sorted]
    except nx.NetworkXError:
        return f"Node {node} does not exist in the graph"

def get_no_connections(graph, node):
    try:
        all_nodes=set(graph.nodes())
        neighbors=set(graph.neighbors(node))

        non_friend=all_nodes-neighbors-{node}

        return list(non_friend)
    except nx.NetworkXError:
        return f"Node {node} does not exist in the graph"
    

st.write("**However, there are still lot more opportunities for collaborations and connections**")
st.write("Below, you can see for each member **full list** of IReNA members that he/she has not collaborated with")

def get_no_connections(graph, node):
    try:
        all_nodes=set(graph.nodes())
        neighbors=set(graph.neighbors(node))

        non_friend=all_nodes-neighbors-{node}

        return list(non_friend)
    except nx.NetworkXError:
        return f"Node {node} does not exist in the graph"




st.write("See below for each member full list of members not yet connected")


node=list(graph_trial.nodes())
options_9=st.selectbox("Choose an IReNA member ID", node)
nodes=get_no_connections(graph_trial, options_9)


if st.button("Apply Queery "):
    st.write(nodes)

    #Creating graph for selected node***********************************************************************
    graph_trial_2=nx.Graph()

    val=data[data["ID"]==options_9]
    Latitude=val["Latitude"].values[0]
    Longitude=val["Longitude"].values[0]

    graph_trial_2.add_node(options_9, Latitude=Latitude, Longitude=Longitude)

    for i in nodes:
        val=data[data["ID"]==i]
        Latitude=val["Latitude"].values[0]
        Longitude=val["Longitude"].values[0]

        graph_trial_2.add_node(i, Latitude=Latitude, Longitude=Longitude)


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

    fig.update_layout(title_text="Location of Possible Connections", title_x=0.5,
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
                        zoom=0.3,
                        style='open-street-map'
                        ),
            margin=dict(t=150)
            )

    st.plotly_chart(fig)

st.write("")
st.markdown("***")
st.write("")

st.header("New Connection Recommendation")

st.write("Current version is for individuals with atleast 1 publication")
node=list(graph_trial.nodes())
option_10=st.selectbox("Choose an IReNA member", node)

tempo_TB_2=get_no_connections(graph_trial, option_10)
main_ind=dict_node_index_2[option_10]

dict_recomm={}
for i in range(len(tempo_TB_2)):
    person=tempo_TB_2[i]
    ind=dict_node_index_2[person]

    with torch.no_grad():
        # Encode the entire graph to get node embeddings
        z = model.encode(data_main.x, data_main.edge_index)

        # Indices of the two nodes you want to predict a link between
        node1_index = main_ind
        node2_index = ind

        # Create the edge index for the new link prediction
        new_edge_index = torch.tensor([[node1_index, node2_index]], device=device).t()

        # Decode to get the link prediction score
        score = model.decode(z, new_edge_index)

    # Convert the score to a probability using sigmoid
    probability = torch.sigmoid(score)
    dict_recomm[person]=probability.item()
    
dict_recomm={k:v for k,v in sorted(dict_recomm.items(), key=lambda item: item[1], reverse=True)}
dict_recomm=dict(list(dict_recomm.items())[:10])
data_2_show=data[data["ID"].isin(dict_recomm.keys())][["First_fix", "Last_fix", "Home Institution", "Network", "Country", "Position"]]
data_2_show["Link Formation Likelihood"]=dict_recomm.values()
st.write(data_2_show.reset_index(drop=True))