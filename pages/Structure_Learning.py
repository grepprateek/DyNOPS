import streamlit as st
import pandas as pd
from datetime import date
import itertools
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, PC, GES
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go

from pyvis.network import Network
import networkx as nx
import tempfile
import streamlit.components.v1 as components
from typing import List
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import json
import numpy as np
import re
import inspect

st.header("Structure Learning")

st.set_page_config(
    layout="wide",  
)

st.html(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #3b766c; 
            color: white;              
            border-color: black;     
        }
        
        div.stButton > button[kind="primary"]:hover {
            background-color: #1B5E20;
            color: white;
        }
        </style>
        """
    )

discretized_dataset_without_delay = pd.read_csv(f"data/dynops_discretized_dataset_without_delay_{date.today().strftime('%d-%m-%y')}.csv")
discretized_dataset_with_delay = pd.read_csv(f"data/dynops_discretized_dataset_with_delay_{date.today().strftime('%d-%m-%y')}.csv")

def visualize_bayesian_network_pyvis(
    edges,
    target_node="full_remission",
    height="850px",
    width="100%"
):
    """
    Visualize Bayesian Network in Streamlit using PyVis.

    Parameters
    ----------
    edges : list[tuple]
        [(parent, child), ...]
    target_node : str
        Outcome node to highlight.
    """

    G = nx.DiGraph()
    G.add_edges_from(edges)

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="black"
    )

    # smoother layout
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.2,
        spring_length=250,
        spring_strength=0.02,
        damping=0.09
    )

    for node in G.nodes():

        if node == target_node:

            color = "#ffadad"
            size = 30

        elif "_(tm" in node:

            color = "#ffd685"
            size = 20

        elif "_(t)" in node:

            color = "#d9ed8f"
            size = 20

        else:

            color = "#dedaf4"
            size = 20

        parents = list(G.predecessors(node))
        children = list(G.successors(node))

        hover_text = (
            f"{node}"
            f"\tParents: {len(parents)}"
            f"\tChildren: {len(children)}"
        )

        net.add_node(
            node,
            label=node,
            title=hover_text,
            color=color,
            size=size,
            borderWidth=2
        )

    for source, target in G.edges():

        net.add_edge(
            source,
            target,
            arrows="to",
            color="#999999"
        )

    # interaction options
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": false,
        "keyboard": true
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html"
    ) as tmp:

        net.save_graph(tmp.name)

        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    st.iframe(
        html,
        height=800
    )


def define_forbidden_edges(
            dataset: pd.DataFrame,
            target_variable: str = None
    ) -> List:
    """
    Creates a list of edges violating the conditions of Dynamic Bayesian Network. 

    Attribute
    ---------
    dataset: pd.DataFrame
    target_variable: str

    Returns
    -------
    forbidden_edges_list: List
    """
    forbidden_edges = []

    # 1. Forbid edges going from target node to other nodes
    if target_variable:
        forbidden_edges.extend([(target_variable, var) for var in dataset.columns if var != target_variable])

    # 2. Forbid edges going backward in time
    forbidden_edges.extend((var1, var2) for var1, var2 in itertools.permutations(dataset.columns, 2)
                            if var1.endswith('(t)') and var2.endswith('(tm1)'))
    forbidden_edges.extend((var1, var2) for var1, var2 in itertools.permutations(dataset.columns, 2)
                        if ('(tm' in var1 and '(tm' in var2) and (int(var1.split('_')[1][3]) < int(var2.split('_')[1][3])))
    return forbidden_edges


def learn_structure(dataset: pd.DataFrame,
        algorithm: str = 'hcs',
        scoring_function: str = 'bic',
        target_variable: str = None
) -> List:
    """
    Performs structure learning on private discrete-valued dataset and returns a Bayesian network edges list.
    
    Attributes
    ----------
    dataset: pd.DataFrame
        private discrete-valued dataset
    algorithm: str
        structure learning algorithm to learn Bayesian network from the dataset. 
        This implementation supports score-based Hill Climb Search`[hcs]`(default), 
        constraint-based Peter-Clark`[pc]`, hybrid Greedy Equivalence Search (GES) algorithm`[ges]`.
    scoring_function: str
        for score-based algorithms, Akaike Information Criterion (AIC)`[aic-d]`, Bayesian Information Criterion (BIC)`[bic-d]` (default),
        and Bayesian Dirichlet Uniform (BDeu)`[bdeu]` are supported.
    target_variable: str
        outcome/target variable to be predicted
    
    Returns
    -------
    bayesian network edges list: List
    """
    dataset = dataset.drop(columns=['Chiffre', 'Date'])
    for col in dataset.columns:
        dataset[col] = dataset[col].astype('category')

    forbidden_edges = define_forbidden_edges(dataset, target_variable)
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

    scores = {
        'bic': 'bic-d',
        'aic': 'aic-d',
        'bdeu': 'bdeu',
        'k2': 'k2'
    }
    score = scores[scoring_function]

    if algorithm == 'hcs':
        model = HillClimbSearch(data=dataset).estimate(scoring_method='bic-d', expert_knowledge=expert_knowledge)
    elif algorithm == 'pc':
        model = PC(data=dataset).estimate(return_type='dag')
    elif algorithm == 'ges':
        model = GES(data=dataset).estimate(scoring_method='bic-d')
    else:
        raise ValueError("Please input a valid algorithm option: 'hcs', 'pc, 'ges'") 

    return dataset, model.edges


def get_timepoint(node):

    match = re.search(r"\((t|tm\d+)\)", node)

    return match.group(1) if match else None


def add_delay_edges(edges, dataset_with_delay):

    delay_cols = {
        col
        for col in dataset_with_delay.columns
        if col.startswith("delay_")
    }

    augmented_edges = list(edges)

    for parent, child in edges:

        parent_tp = get_timepoint(parent)
        child_tp = get_timepoint(child)

        # only consider temporal nodes
        if parent_tp is None or child_tp is None:
            continue

        # skip same-time arcs
        if parent_tp == child_tp:
            continue

        delay_name = f"delay_{parent_tp}_{child_tp}"

        if delay_name not in delay_cols:
            delay_name = f"delay_{child_tp}_{parent_tp}"

        if delay_name in delay_cols:
            augmented_edges.append(
                (delay_name, child)
            )

    return augmented_edges


def visualize_bayesian_network(edges):
    G = nx.DiGraph(edges)
    pos = nx.spring_layout(G, seed=23, k=2.5)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # 4. Extract Node Coordinates for Plotly
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Label names on hover or display

    # Create the node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # Shows both the circle marker and the text label
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color="#1f77b4",  # Nice custom blue color
            size=25,
            line=dict(width=2, color="white"),
        ),
    )

    # 5. Build and Layout the Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Bayesian Network Structure",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",  # Clean backdrop
        ),
    )

    return fig

with st.expander("View documentation"):
    st.markdown("""
        **Algorithm:** Hill Climb Search\n
        **Scoring method:** Bayesian Information Criterion (sparse networks)
    """)

with st.expander("View source code"):
    code = "\n\n".join([
        inspect.getsource(learn_structure),
        inspect.getsource(define_forbidden_edges),
        inspect.getsource(add_delay_edges),
    ])
    st.code(code, language='python')
    

def create_network_dict(edges, dataset_columns):
    parents = defaultdict(list)

    for parent, child in edges:
        parents[child].append(parent)

    return {
        node: list(parents[node])
        for node in dataset_columns
    }

def compute_beta_params(dataset: pd.DataFrame, network_edges: dict):
    """
    Computes the multi-logit regression parameters for each node in the given network using the dataset. 
    If a node does not have any parents, the value counts for each category are normalized to obtain probability for each class.
    """
    network_params = {}
    for variable in dataset.columns:
        parents = network_edges[variable]
        if not parents:
            probs = dataset[variable].value_counts(normalize=True).sort_index().to_dict()
            network_params[variable] = {
                "has_parents": False,
                "params": probs
            }
        
        else:
            X = pd.get_dummies(dataset[parents], drop_first=True, dtype=int)
            X = X.loc[:, ~X.columns.duplicated()]   # remove duplicate columns
            X.insert(0, "Intercept", 1)
            y = pd.Categorical(dataset[variable]).codes
            lr = LogisticRegression(solver="lbfgs", fit_intercept=False, max_iter=500)
            lr.fit(X, y)
            network_params[variable] = {
                "has_parents": True,
                "parents": network_edges[variable],
                "params": lr.coef_.tolist()
            }

    return network_params



if st.button("Learn DBN structure", type="primary"):
    with st.spinner("Please wait...good things and structure learning take time!"):
        structure_dataset, edges = learn_structure(
            dataset=discretized_dataset_without_delay,
            algorithm="hcs",
            target_variable="full_remission"
        )

        edges = add_delay_edges(
            edges,
            discretized_dataset_with_delay
        )

        visualize_bayesian_network_pyvis(edges)

    parameter_dataset = (
        discretized_dataset_with_delay
        .drop(columns=["Chiffre", "Date"])
        .copy()
    )

    network_dict = create_network_dict(
        edges,
        parameter_dataset.columns
    )

    params = compute_beta_params(
        parameter_dataset,
        network_dict
    )

    with open(
        "network_parameters.json",
        "w"
    ) as pf:

        json.dump(
            params,
            pf,
            indent=4
        )

    st.success(
        f"Learned {len(edges)} edges."
    )

    st.write(
        "Number of nodes:",
        len(network_dict)
    )


if st.button("Next: Evaluation", type='primary'):
    st.switch_page("pages/Evaluation.py")
