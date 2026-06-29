import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import date
from typing import Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix,
)
import plotly.graph_objects as go

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")

st.html("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #3b766c; color: white; border-color: black;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1B5E20; color: white;
}
</style>
""")

st.header("DBN Comparison")
st.write("Upload two `network_parameters.json` files to compare their structure and predictive performance.")

# ── colours ───────────────────────────────────────────────────────────────────
C1 = "#3b766c"   # teal  – DBN A
C2 = "#c0392b"   # red   – DBN B

def _rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── prediction helpers ────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_probability(test_df: pd.DataFrame, node: str, params: dict) -> np.ndarray:
    info = params[node]
    if not info["has_parents"]:
        p1 = float(info["params"].get("1", info["params"].get(1, 0.0)))
        return np.full(len(test_df), p1)
    parents = info["parents"]
    coef_matrix = np.array(info["params"])
    X = pd.get_dummies(test_df[parents].copy(), drop_first=True, dtype=int).values.astype(float)
    X_full = np.hstack([np.ones((X.shape[0], 1)), X])
    logits_pos = X_full @ coef_matrix.T
    if coef_matrix.shape[0] == 1:
        return 1.0 / (1.0 + np.exp(-logits_pos[:, 0]))
    logits_all = np.hstack([np.zeros((X_full.shape[0], 1)), logits_pos])
    return _softmax(logits_all)[:, 1]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
        "PR-AUC":    average_precision_score(y_true, y_prob),
        "Log Loss":  log_loss(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }

HIGHER_BETTER = {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"}
LOWER_BETTER  = {"Log Loss", "Brier"}


# ── structure analysis helpers ────────────────────────────────────────────────

def _is_temporal(node: str) -> bool:
    return bool(re.search(r"\((t|tm\d+)\)", node))

def _is_inter_slice(parent: str, child: str) -> bool:
    """tm* → t edge."""
    return bool(re.search(r"\(tm\d+\)", parent)) and bool(re.search(r"\(t\)", child))

def _slice_tag(node: str) -> str:
    m = re.search(r"\((t|tm\d+)\)", node)
    return m.group(1) if m else "static"

def extract_edges(params: dict) -> list[tuple[str, str]]:
    edges = []
    for child, info in params.items():
        if info["has_parents"]:
            for parent in info["parents"]:
                edges.append((parent, child))
    return edges

def structure_summary(params: dict) -> dict:
    edges = extract_edges(params)
    nodes = set(params.keys())
    inter = [(p, c) for p, c in edges if _is_inter_slice(p, c)]
    intra = [(p, c) for p, c in edges if not _is_inter_slice(p, c)]
    no_parents = [n for n, v in params.items() if not v["has_parents"]]
    max_parents = max((len(v["parents"]) for v in params.values() if v["has_parents"]), default=0)
    most_connected = max(params.items(), key=lambda kv: len(kv[1].get("parents", [])))
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "inter_slice_edges": len(inter),
        "intra_slice_edges": len(intra),
        "root_nodes": len(no_parents),
        "max_parents": max_parents,
        "most_connected_node": most_connected[0],
        "most_connected_parents": len(most_connected[1].get("parents", [])),
        "target_parents": params.get("full_remission", {}).get("parents", []),
        "edge_list": edges,
        "inter_edges": inter,
    }


# ── chart helpers ─────────────────────────────────────────────────────────────

METRIC_LIST = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Log Loss", "Brier"]

def _metric_bar_chart(m1: dict, m2: dict, label1: str, label2: str) -> go.Figure:
    metrics = METRIC_LIST
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=label1, x=metrics, y=[m1[m] for m in metrics],
        marker_color=C1, marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name=label2, x=metrics, y=[m2[m] for m in metrics],
        marker_color=C2, marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, range=[0, 1.05], title="Score"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=20, l=40, r=10),
        height=340,
    )
    return fig


def _radar_chart(m1: dict, m2: dict, label1: str, label2: str) -> go.Figure:
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    theta = radar_metrics + radar_metrics[:1]
    fig = go.Figure()
    for m, color, label in [(m1, C1, label1), (m2, C2, label2)]:
        vals = [m[k] for k in radar_metrics] + [m[radar_metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, fill="toself", name=label,
            line_color=color, fillcolor=_rgba(color, 0.15),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#ddd")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="#ffffff",
        margin=dict(t=20, b=60, l=40, r=40),
        height=380,
    )
    return fig


def _prob_dist_chart(p1: np.ndarray, p2: np.ndarray, label1: str, label2: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=p1, name=label1, marker_color=_rgba(C1, 0.7),
        nbinsx=20, opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=p2, name=label2, marker_color=_rgba(C2, 0.7),
        nbinsx=20, opacity=0.85,
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Predicted P(remission=1) distribution", font_size=14, x=0),
        xaxis_title="Predicted probability", yaxis_title="Count",
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=30, l=40, r=10),
        height=300,
    )
    return fig


def _confusion_heatmap(y_true: np.ndarray, y_prob: np.ndarray, label: str, color: str) -> go.Figure:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0, "#ffffff"], [1, color]],
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {label}", font_size=13, x=0),
        margin=dict(t=40, b=20, l=10, r=10),
        height=260,
        paper_bgcolor="#ffffff",
    )
    return fig


def _edge_overlap_chart(edges1: list, edges2: list, label1: str, label2: str) -> go.Figure:
    s1, s2 = set(edges1), set(edges2)
    only1  = len(s1 - s2)
    shared = len(s1 & s2)
    only2  = len(s2 - s1)
    fig = go.Figure(go.Bar(
        x=[f"Only in {label1}", "Shared", f"Only in {label2}"],
        y=[only1, shared, only2],
        marker_color=[C1, "#888888", C2],
        marker_line_width=0,
        text=[only1, shared, only2],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Edge overlap between the two networks", font_size=14, x=0),
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, title="# edges"),
        xaxis=dict(showgrid=False),
        margin=dict(t=40, b=20, l=40, r=10),
        height=300,
    )
    return fig


# ── metric card HTML ──────────────────────────────────────────────────────────

def _metric_card(metric: str, v1: float, v2: float, label1: str, label2: str) -> str:
    higher = metric in HIGHER_BETTER
    winner = label1 if (v1 > v2 if higher else v1 < v2) else label2
    w_color = C1 if winner == label1 else C2
    arrow = "↑" if higher else "↓"
    return f"""
<div style="border:1.5px solid #e2e8f0;border-radius:10px;padding:14px 12px;
            background:#fafafa;text-align:center;height:100%;">
  <div style="font-size:12px;font-weight:600;color:#555;letter-spacing:.05em;
              text-transform:uppercase;margin-bottom:6px">{metric}
    <span style="font-weight:400;color:#999">{arrow} {'higher' if higher else 'lower'}</span>
  </div>
  <div style="display:flex;justify-content:space-around;align-items:center;gap:8px">
    <div>
      <div style="font-size:22px;font-weight:700;color:{C1}">{v1:.4f}</div>
      <div style="font-size:11px;color:#777">{label1}</div>
    </div>
    <div style="font-size:18px;color:#bbb">vs</div>
    <div>
      <div style="font-size:22px;font-weight:700;color:{C2}">{v2:.4f}</div>
      <div style="font-size:11px;color:#777">{label2}</div>
    </div>
  </div>
  <div style="margin-top:8px;font-size:11px;color:{w_color};font-weight:600">
    {winner} wins
  </div>
</div>
"""


# ── main UI ───────────────────────────────────────────────────────────────────

with st.expander("Read documentation — metrics explained"):
    st.markdown("""
| Metric | What it measures |
|---|---|
| **Accuracy** | Share of all predictions (both remission and non-remission) that are correct. |
| **Precision** | Of all patients predicted to remit, the fraction who actually did — penalises false alarms. |
| **Recall** | Of all patients who actually remitted, the fraction the model correctly identified — penalises missed cases. |
| **F1** | Harmonic mean of Precision and Recall; useful when the two need to be balanced together. |
| **ROC-AUC** | Probability that the model ranks a random remission case higher than a random non-remission case across all thresholds. |
| **PR-AUC** | Area under the Precision-Recall curve; more informative than ROC-AUC when the positive class is rare. |
| **Log Loss** | Average log-probability assigned to the true label; penalises confident wrong predictions — lower is better. |
| **Brier Score** | Mean squared error between predicted probabilities and true binary outcomes — lower is better. |
""")

st.divider()

# ── file uploads ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"#### <span style='color:{C1}'>DBN A</span>", unsafe_allow_html=True)
    file1   = st.file_uploader("Upload network_parameters.json", key="f1", type="json")
    label1  = st.text_input("Label", value="DBN A", key="l1")

with col_b:
    st.markdown(f"#### <span style='color:{C2}'>DBN B</span>", unsafe_allow_html=True)
    file2   = st.file_uploader("Upload network_parameters.json", key="f2", type="json")
    label2  = st.text_input("Label", value="DBN B", key="l2")

if not (file1 and file2):
    st.info("Upload both JSON files above to begin the comparison.")
    st.stop()

params1 = json.load(file1)
params2 = json.load(file2)

# ── ① structure expanders ─────────────────────────────────────────────────────
st.divider()
st.subheader("Network Structure")

s1 = structure_summary(params1)
s2 = structure_summary(params2)

col_a, col_b = st.columns(2)

for col, s, label, color in [(col_a, s1, label1, C1), (col_b, s2, label2, C2)]:
    with col:
        with st.expander(f"{label} — structure details", expanded=True):
            st.markdown(f"""
<div style="border-left:4px solid {color};padding-left:12px;margin-bottom:10px">
  <b>Nodes:</b> {s['nodes']} &nbsp;|&nbsp;
  <b>Edges:</b> {s['edges']} &nbsp;|&nbsp;
  <b>Inter-slice:</b> {s['inter_slice_edges']} &nbsp;|&nbsp;
  <b>Intra-slice:</b> {s['intra_slice_edges']}<br>
  <b>Root nodes (no parents):</b> {s['root_nodes']} &nbsp;|&nbsp;
  <b>Max parents:</b> {s['max_parents']}<br>
  <b>Most connected:</b> {s['most_connected_node']} ({s['most_connected_parents']} parents)
</div>
""", unsafe_allow_html=True)

            st.markdown("**Parents of `full_remission`:**")
            if s["target_parents"]:
                for p in s["target_parents"]:
                    tag = _slice_tag(p)
                    badge_color = "#ffd685" if tag.startswith("tm") else "#d9ed8f" if tag == "t" else "#dedaf4"
                    st.markdown(
                        f"<span style='background:{badge_color};padding:2px 8px;"
                        f"border-radius:4px;font-size:13px'>{p}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_No parents learned_")

            st.markdown("**All edges:**")
            edge_df = pd.DataFrame(s["edge_list"], columns=["Parent", "Child"])
            edge_df["Type"] = edge_df.apply(
                lambda r: "inter-slice" if _is_inter_slice(r.Parent, r.Child) else "intra-slice", axis=1
            )
            st.dataframe(edge_df, use_container_width=True, height=260)


# Edges only in one network
s1_set, s2_set = set(s1["edge_list"]), set(s2["edge_list"])
col_a, col_b = st.columns(2)
with col_a:
    with st.expander(f"Edges only in {label1}"):
        only1 = sorted(s1_set - s2_set)
        if only1:
            st.dataframe(pd.DataFrame(only1, columns=["Parent", "Child"]), use_container_width=True)
        else:
            st.markdown("_None_")
with col_b:
    with st.expander(f"Edges only in {label2}"):
        only2 = sorted(s2_set - s1_set)
        if only2:
            st.dataframe(pd.DataFrame(only2, columns=["Parent", "Child"]), use_container_width=True)
        else:
            st.markdown("_None_")

with st.expander("Shared edges (in both networks)"):
    shared = sorted(s1_set & s2_set)
    if shared:
        st.dataframe(pd.DataFrame(shared, columns=["Parent", "Child"]), use_container_width=True)
    else:
        st.markdown("_No shared edges_")

# ── ② evaluation ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Evaluation")

data_file = st.file_uploader(
    "Upload the test dataset (CSV with `full_remission` column and `Chiffre` for grouping)",
    type="csv",
    key="data",
)

if not data_file:
    st.info("Upload the dataset CSV above to evaluate both networks.")
    st.stop()

test_csv = pd.read_csv(data_file)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
_, test_idx = next(splitter.split(test_csv, groups=test_csv["Chiffre"]))
test_df = test_csv.iloc[test_idx].copy()

y_true = test_df["full_remission"].astype(int).values

with st.spinner("Computing predictions…"):
    try:
        y_prob1 = predict_probability(test_df, "full_remission", params1)
        y_prob2 = predict_probability(test_df, "full_remission", params2)
    except KeyError as e:
        st.error(f"Column missing in test data: {e}. Make sure the dataset matches the network variables.")
        st.stop()

m1 = evaluate(y_true, y_prob1)
m2 = evaluate(y_true, y_prob2)

with st.expander("Metric-by-metric comparison", expanded=True):
    # 4-per-row metric cards
    metrics_rows = [METRIC_LIST[:4], METRIC_LIST[4:]]
    for row in metrics_rows:
        cols = st.columns(4)
        for col, metric in zip(cols, row):
            with col:
                st.markdown(_metric_card(metric, m1[metric], m2[metric], label1, label2),
                            unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

with st.expander("Radar — overall profile"):
    st.plotly_chart(_radar_chart(m1, m2, label1, label2), use_container_width=True)

with st.expander("Grouped bar — all metrics"):
    st.plotly_chart(_metric_bar_chart(m1, m2, label1, label2), use_container_width=True)

with st.expander("Confusion matrices"):
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(_confusion_heatmap(y_true, y_prob1, label1, C1), use_container_width=True)
    col_b.plotly_chart(_confusion_heatmap(y_true, y_prob2, label2, C2), use_container_width=True)

with st.expander("Predicted probability distributions"):
    st.plotly_chart(_prob_dist_chart(y_prob1, y_prob2, label1, label2), use_container_width=True)

with st.expander("Raw metrics table"):
    cmp_df = pd.DataFrame({"Metric": METRIC_LIST,
                           label1: [m1[m] for m in METRIC_LIST],
                           label2: [m2[m] for m in METRIC_LIST]})
    cmp_df["Winner"] = cmp_df.apply(
        lambda r: label1 if (r[label1] > r[label2] if r.Metric in HIGHER_BETTER else r[label1] < r[label2])
        else label2, axis=1
    )
    st.dataframe(cmp_df.set_index("Metric").style.format({label1: "{:.4f}", label2: "{:.4f}"}),
                 use_container_width=True)

if st.button("Back: Baseline Comparison", type="primary"):
    st.switch_page("pages/Baselines_Comparison.py")