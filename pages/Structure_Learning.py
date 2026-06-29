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

def _base_name(node: str) -> str:
    """Strip any time-slice suffix to get the variable base name.

    Examples
    --------
    'age_(t)'    → 'age'
    'age_(t+1)'  → 'age'
    'age_(tm1)'  → 'age'
    'age'        → 'age'   (static, unchanged)
    """
    return re.sub(r"_\((t\+?\d*|tm\d*)\)$", "", node)


def visualize_bayesian_network_pyvis(
    edges,
    target_node: str = "full_remission",
):
    """
    Visualize a Dynamic Bayesian Network as a two-slice (t) / (t+1) layout.

    • All nodes — including static variables — appear in both slice replicas.
    • Layout: spring_layout on the (t) slice, mirrored into the (t+1) column.
    • Rendering: generate_html (pyvis ≥ 0.3.x) with save_graph fallback.
    """

    # ------------------------------------------------------------------ #
    # 0.  Node classification helpers                                     #
    # ------------------------------------------------------------------ #
    def is_t(node):    return "_(t)" in node and "_(t+" not in node
    def is_t1(node):   return "_(t+1)" in node
    def is_tm(node):   return bool(re.search(r"_\(tm\d+\)", node))
    def is_delay(node): return node.startswith("delay_")

    def is_static(node):
        return not is_t(node) and not is_t1(node) and not is_tm(node) and not is_delay(node)

    # ------------------------------------------------------------------ #
    # 1.  Normalise input edges to (t) / (t+1) naming                    #
    # ------------------------------------------------------------------ #
    # Collect all raw nodes to find static bases
    all_raw = set()
    for src, dst in edges:
        all_raw.add(src)
        all_raw.add(dst)

    static_bases = sorted(n for n in all_raw if is_static(n))

    def to_t(node):
        """Return the (t) version of a node."""
        if is_tm(node):
            # tm1 → t,  tm2 → t  (past slice maps to left / t side)
            return re.sub(r"_\(tm\d+\)$", "_(t)", node)
        if is_static(node):
            return f"{node}_(t)"
        return node   # already (t) or (t+1)

    def to_t1(node):
        """Return the (t+1) version of a node."""
        if is_tm(node):
            return re.sub(r"_\(tm\d+\)$", "_(t+1)", node)
        if is_t(node):
            return node.replace("_(t)", "_(t+1)")
        if is_static(node):
            return f"{node}_(t+1)"
        return node

    # Build expanded edge list in (t)/(t+1) space.
    # Rule: only edges that structure learning produced — no fabrication.
    #
    #   tm* → t/static  (inter-slice)  →  base_(t) → base_(t+1)
    #   t/static → t/static (intra)    →  mirrored into both slices
    #   tm* → tm*           (intra)    →  mirrored into both slices
    expanded = []
    for src, dst in edges:
        if is_delay(src) or is_delay(dst):
            continue   # delay edges excluded from visualisation

        s_is_past = is_tm(src)
        d_is_past = is_tm(dst)
        s_is_curr = is_t(src)
        d_is_curr = is_t(dst)
        s_is_sta  = is_static(src)
        d_is_sta  = is_static(dst)

        if s_is_past and (d_is_curr or d_is_sta):
            # Inter-slice: tm* → t  →  base_(t) → base_(t+1)
            expanded.append((to_t(src), to_t1(dst)))

        elif (s_is_curr or s_is_sta) and (d_is_curr or d_is_sta):
            # Intra-slice: both on the current (t) side — mirror into both slices
            expanded.append((to_t(src),  to_t(dst)))
            expanded.append((to_t1(src), to_t1(dst)))

        elif s_is_past and d_is_past:
            # Intra-past: both tm* — mirror into both slices
            expanded.append((to_t(src),  to_t(dst)))
            expanded.append((to_t1(src), to_t1(dst)))

        else:
            expanded.append((src, dst))

    # Deduplicate exact copies
    seen: set = set()
    deduped = []
    for u, v in expanded:
        if (u, v) not in seen:
            seen.add((u, v))
            deduped.append((u, v))
    expanded = deduped

    # ------------------------------------------------------------------ #
    # Enforce DAG on each slice replica                                   #
    # ------------------------------------------------------------------ #
    # Collect intra-slice edges for (t) and (t+1) separately
    raw_intra_t  = [(u, v) for u, v in expanded if is_t(u)  and is_t(v)]
    raw_intra_t1 = [(u, v) for u, v in expanded if is_t1(u) and is_t1(v)]

    def dag_edges(edge_list):
        """
        Given a list of directed edges, return a subset that forms a DAG.
        Build a topological order on the undirected skeleton, then keep only
        edges whose direction agrees with that order.  Any edge whose reverse
        is also present (cycle of length 2) gets resolved this way without
        losing either node.
        """
        G = nx.DiGraph(edge_list)
        if nx.is_directed_acyclic_graph(G):
            return edge_list          # already fine — return as-is

        # Get an undirected topological hint via longest-path ordering
        # on the DAG condensation (collapses SCCs)
        condensation = nx.condensation(G)
        # Map each node to its SCC representative's topo index
        topo_index = {n: i for i, n in enumerate(nx.topological_sort(condensation))}
        node_rank  = {
            node: topo_index[data["members"].__iter__().__next__()
                             if False else scc_id]
            for scc_id, data in condensation.nodes(data=True)
            for node in data["members"]
        }
        # Keep edge u→v only if rank[u] <= rank[v]; drop the reverse
        kept = []
        kept_set = set()
        for u, v in edge_list:
            if (v, u) in kept_set:
                continue   # reverse already kept — skip this one
            kept.append((u, v))
            kept_set.add((u, v))
        return kept

    intra_t_dag  = dag_edges(raw_intra_t)
    intra_t1_dag = dag_edges(raw_intra_t1)

    # Rebuild expanded with DAG-enforced intra-slice edges
    inter_edges  = [(u, v) for u, v in expanded if is_t(u) and is_t1(v)]
    expanded = intra_t_dag + intra_t1_dag + inter_edges

    # ------------------------------------------------------------------ #
    # 2.  Separate into intra / inter / delay buckets                     #
    # ------------------------------------------------------------------ #
    all_nodes = set()
    for u, v in expanded:
        all_nodes.add(u); all_nodes.add(v)

    t_nodes  = sorted(n for n in all_nodes if is_t(n))
    t1_nodes = sorted(n for n in all_nodes if is_t1(n))
    dly_nodes = sorted(n for n in all_nodes if is_delay(n))

    intra_t  = [(u, v) for u, v in expanded if is_t(u)  and is_t(v)]
    intra_t1 = [(u, v) for u, v in expanded if is_t1(u) and is_t1(v)]
    inter    = [(u, v) for u, v in expanded if is_t(u)  and is_t1(v)]
    delay_e  = [(u, v) for u, v in expanded if is_delay(u) or is_delay(v)]

    # ------------------------------------------------------------------ #
    # 3.  Positions — spring on (t) slice, mirror for (t+1)              #
    # ------------------------------------------------------------------ #
    G_t = nx.DiGraph()
    G_t.add_nodes_from(t_nodes)
    G_t.add_edges_from(intra_t)

    pos_t = nx.spring_layout(G_t, k=10, iterations=300, seed=23)

    X_OFFSET = 3.5
    X_SCALE  = 3
    Y_SCALE  = 5

    pos = {}
    for node, (x, y) in pos_t.items():
        base = _base_name(node)
        pos[f"{base}_(t)"]   = (x * X_SCALE - X_OFFSET, y * Y_SCALE)
        pos[f"{base}_(t+1)"] = (x * X_SCALE + X_OFFSET, y * Y_SCALE)

    # Delay nodes — centre strip
    for i, node in enumerate(dly_nodes):
        pos[node] = (0, -3 + i * 1.5)

    for node in all_nodes:
        if node not in pos:
            pos[node] = (0, 0)

    # ------------------------------------------------------------------ #
    # 4.  Build PyVis network                                             #
    # ------------------------------------------------------------------ #
    try:
        net = Network(
            height="1000px",
            width="1800px",
            directed=True,
            bgcolor="#ffffff",
            cdn_resources="in_line",
        )
    except TypeError:
        net = Network(
            height="1000px",
            width="1800px",
            directed=True,
            bgcolor="#ffffff",
        )

    G_full = nx.DiGraph()
    G_full.add_edges_from(expanded)

    # ── colour palette ─────────────────────────────────────────────── #
    C_T_FILL     = "#a8d8f0"   # sky-blue   – (t) temporal nodes
    C_T_BORDER   = "black"
    C_T1_FILL    = "#a8e6b8"   # mint       – (t+1) temporal nodes
    C_T1_BORDER  = "black"
    C_TGT_FILL   = "#ffadad"   # rose       – target node
    C_TGT_BORDER = "black"
    C_STA_FILL   = "#ffd6a5"   # amber      – static (no time suffix)
    C_STA_BORDER = "black"

    for node, (x, y) in pos.items():
        if node not in all_nodes:
            continue
        label = _base_name(node)
        is_sta = label in static_bases   # originally had no time suffix

        if label == target_node:
            fill, border = C_TGT_FILL, C_TGT_BORDER
        elif is_sta:
            fill, border = C_STA_FILL, C_STA_BORDER
        elif is_t1(node):
            fill, border = C_T1_FILL, C_T1_BORDER
        else:
            fill, border = C_T_FILL, C_T_BORDER

        net.add_node(
            node,
            label=label,
            x=x * 400,
            y=y * 250,
            physics=False,
            color={
                "background": fill,
                "border":     border,
                "highlight":  {"background": fill, "border": border},
                "hover":      {"background": fill, "border": border},
            },
            size=50,
            borderWidth=2,
            font={"size": 30, "color": "#111111"},
        )

    # ── edges — thicker, cleaner ───────────────────────────────────── #
    for u, v in intra_t:
        net.add_edge(u, v, color="rgba(0,0,0,0.45)", width=4)
    for u, v in intra_t1:
        net.add_edge(u, v, color="rgba(0,0,0,0.45)", width=4)
    for u, v in inter:
        net.add_edge(u, v, color="#a888b5", width=5, dashes=True)

    net.set_options("""
var options = {
  "physics": { "enabled": false },
  "edges": {
    "smooth": { "type": "continuous", "roundness": 0.15 },
    "arrows": { "to": { "enabled": true, "scaleFactor": 1.2 } }
  }
}
""")

    # ------------------------------------------------------------------ #
    # 5.  HTML injection                                                  #
    # ------------------------------------------------------------------ #
    # Single script: inject slice-title overlay AFTER vis.js canvas is ready,
    # then fit the network. Labels live in a sibling div above the canvas.
    inject_script = """
<script>
(function waitForVis() {
  if (typeof network === "undefined") { setTimeout(waitForVis, 100); return; }

  // Fit whole graph on first render
  network.fit({ animation: false });

  // Build a sibling overlay div that floats above the vis canvas
  var wrap = document.getElementById("mynetwork");
  if (!wrap) return;

  // Make sure the parent is positioned so our overlay can anchor to it
  wrap.style.position = "relative";

  var overlay = document.createElement("div");
  overlay.style.cssText = [
    "position:absolute",
    "top:12px",
    "left:0",
    "width:100%",
    "display:flex",
    "pointer-events:none",
    "z-index:9999",
    "box-sizing:border-box",
  ].join(";");

  overlay.innerHTML =
    "<div style='width:50%;text-align:center;" +
    "font-family:Segoe UI,sans-serif;font-size:20px;font-weight:700;" +
    "color:#1a5f8a;letter-spacing:0.02em;" +
    "background:rgba(168,216,240,0.55);border-radius:8px;" +
    "margin:0 6px;padding:5px 0;border:1.5px solid rgba(26,122,181,0.35);'>" +
    "Slice (t)</div>" +
    "<div style='width:50%;text-align:center;" +
    "font-family:Segoe UI,sans-serif;font-size:20px;font-weight:700;" +
    "color:#1a6e36;letter-spacing:0.02em;" +
    "background:rgba(168,230,184,0.55);border-radius:8px;" +
    "margin:0 6px;padding:5px 0;border:1.5px solid rgba(26,140,69,0.35);'>" +
    "Slice (t+1)</div>";

  wrap.appendChild(overlay);
})();
</script>
"""

    try:
        html = net.generate_html()
    except AttributeError:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                html = f.read()

    # Remove default body margin/padding, fix network div to full width
    html = html.replace(
        "<body>",
        "<body style='margin:0;padding:0;overflow:hidden;'>",
    )
    html = html.replace(
        "#mynetwork {",
        "#mynetwork { width:100% !important; position:relative; margin:0; padding:0;",
    )

    # Title: centred above the canvas, zero top padding
    title_bar = (
        "<div style='"
        "width:100%;text-align:center;"
        "font-family:Segoe UI,sans-serif;font-size:28px;font-weight:700;"
        "padding:6px 0 4px 0;margin:0;color:#222;box-sizing:border-box;"
        "'>DyNOPS Dynamic Bayesian Network</div>\n"
    )

    html = html.replace('<div id="mynetwork"', title_bar + '<div id="mynetwork"')
    html = html.replace("</body>", inject_script + "\n</body>")

    components.html(html, height=1060, scrolling=False)


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
    pos = nx.spring_layout(G, seed=23, k=10)

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
            size=40,
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
            dummies = []
            for parent in parents:
                d = pd.get_dummies(dataset[parent], prefix=parent, drop_first=True, dtype=int)
                dummies.append(d)
            X = pd.concat(dummies, axis=1)

            # Add intercept column
            X.insert(0, "Intercept", 1)

            # Convert to numpy to avoid narwhals duplicate column name validation
            X_array = X.to_numpy()

            y = pd.Categorical(dataset[variable]).codes
            lr = LogisticRegression(solver="lbfgs", fit_intercept=False, max_iter=500)
            lr.fit(X_array, y)

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

        viz_edges = [(u, v) for u, v in edges if not u.startswith("delay_") and not v.startswith("delay_")]
        visualize_bayesian_network_pyvis(viz_edges)

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
        f"Learned {len(edges)} edges for {len(network_dict)} nodes."
    )

    st.write(
        "Number of nodes:",
        len(network_dict)
    )


if st.button("Next: Evaluation", type='primary'):
    st.switch_page("pages/Evaluation.py")
