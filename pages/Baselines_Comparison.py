import streamlit as st
import pandas as pd
import numpy as np
import itertools
import json
from datetime import date
from collections import defaultdict
from typing import List

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
)
import plotly.graph_objects as go
import plotly.express as px
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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

st.header("Model Comparison")
st.caption("5-fold group cross-validation — DBN vs. baseline classifiers")

# ── colour palette (one per model, consistent everywhere) ────────────────────
MODEL_COLORS = {
    "DBN":               "#3b766c",
    "LogisticRegression":"#4e8bc4",
    "RandomForest":      "#e07b39",
    "XGBoost":           "#9b59b6",
}

# ── helper functions ──────────────────────────────────────────────────────────

def define_forbidden_edges(dataset: pd.DataFrame, target_variable: str = None) -> List:
    forbidden = []
    if target_variable:
        forbidden.extend([(target_variable, v) for v in dataset.columns if v != target_variable])
    forbidden.extend(
        (v1, v2) for v1, v2 in itertools.permutations(dataset.columns, 2)
        if v1.endswith("(t)") and v2.endswith("(tm1)")
    )
    forbidden.extend(
        (v1, v2) for v1, v2 in itertools.permutations(dataset.columns, 2)
        if ("(tm" in v1 and "(tm" in v2)
        and (int(v1.split("_")[1][3]) < int(v2.split("_")[1][3]))
    )
    return forbidden


def learn_structure(dataset: pd.DataFrame, target_variable: str = None):
    dataset = dataset.drop(columns=["Chiffre", "Date"])
    for col in dataset.columns:
        dataset[col] = dataset[col].astype("category")
    forbidden = define_forbidden_edges(dataset, target_variable)
    ek = ExpertKnowledge(forbidden_edges=forbidden)
    model = HillClimbSearch(data=dataset).estimate(scoring_method="bic-d", expert_knowledge=ek)
    return dataset, model.edges


def create_network_dict(edges, columns) -> dict:
    parents = defaultdict(list)
    for p, c in edges:
        parents[c].append(p)
    return {node: list(parents[node]) for node in columns}


def compute_beta_params(dataset: pd.DataFrame, network_edges: dict) -> dict:
    params = {}
    for var in dataset.columns:
        pars = network_edges[var]
        if not pars:
            params[var] = {
                "has_parents": False,
                "params": dataset[var].value_counts(normalize=True).sort_index().to_dict(),
            }
        else:
            for p in pars:
                dataset[p] = dataset[p].astype("category")
            parent_levels = {p: dataset[p].cat.categories.tolist() for p in pars}
            X = pd.get_dummies(dataset[pars], drop_first=True, dtype=int)
            y = pd.Categorical(dataset[var]).codes
            lr = LogisticRegression(solver="lbfgs", fit_intercept=True, max_iter=500)
            lr.fit(X, y)
            params[var] = {
                "has_parents": True,
                "parents": pars,
                "model": lr,
                "feature_names": X.columns.tolist(),
                "parent_levels": parent_levels,
            }
    return params


def predict_node_probability(test_df: pd.DataFrame, node: str, network_params: dict) -> np.ndarray:
    info = network_params[node]
    if not info["has_parents"]:
        return np.full(len(test_df), float(info["params"].get(1, info["params"].get("1", 0.0))))
    pars = info["parents"]
    X_src = test_df[pars].copy()
    for p in pars:
        X_src[p] = pd.Categorical(X_src[p], categories=info["parent_levels"][p])
    X_test = pd.get_dummies(X_src, drop_first=True, dtype=int)
    X_test = X_test.reindex(columns=info["feature_names"], fill_value=0)
    return info["model"].predict_proba(X_test)[:, 1]


def evaluate_predictions(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_prob),
        "PR_AUC":    average_precision_score(y_true, y_prob),
        "LogLoss":   log_loss(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }


def build_baseline_features(train_df, test_df):
    drop_cols = ["Chiffre", "Date", "full_remission"]
    X_train = pd.get_dummies(train_df.drop(columns=drop_cols), drop_first=True)
    X_test  = pd.get_dummies(test_df.drop(columns=drop_cols),  drop_first=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train = train_df["full_remission"].astype(int)
    y_test  = test_df["full_remission"].astype(int)
    return X_train, X_test, y_train, y_test


# ── chart helpers ─────────────────────────────────────────────────────────────

METRICS_HIGHER = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
METRICS_LOWER  = ["LogLoss", "Brier"]
ALL_METRICS    = METRICS_HIGHER + METRICS_LOWER
METRIC_LABELS  = {
    "Accuracy": "Accuracy", "Precision": "Precision", "Recall": "Recall",
    "F1": "F1", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC",
    "LogLoss": "Log Loss", "Brier": "Brier Score",
}

def _bar_chart(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for _, row in summary.iterrows():
        model = row["Model"]
        fig.add_trace(go.Bar(
            name=model,
            x=[model],
            y=[row["Mean"]],
            error_y=dict(type="data", array=[row["Std"]], visible=True, thickness=2, width=6),
            marker_color=MODEL_COLORS.get(model, "#888"),
            marker_line_width=0,
            width=0.45,
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a1a2e"), x=0),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, title=METRIC_LABELS[metric]),
        xaxis=dict(showgrid=False),
        margin=dict(t=40, b=20, l=40, r=10),
        height=300,
    )
    return fig


def _grouped_bar(summary_df: pd.DataFrame, metrics: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for model, grp in summary_df.groupby("Model"):
        means = [grp.loc[grp["Metric"] == m, "Mean"].values[0] for m in metrics]
        fig.add_trace(go.Bar(
            name=model,
            x=[METRIC_LABELS[m] for m in metrics],
            y=means,
            marker_color=MODEL_COLORS.get(model, "#888"),
            marker_line_width=0,
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=15, color="#1a1a2e"), x=0),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, range=[0, 1.05]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=20, l=40, r=10),
        height=360,
    )
    return fig


def _line_chart(results_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for model, grp in results_df.groupby("Model"):
        fig.add_trace(go.Scatter(
            x=grp["Fold"], y=grp[metric],
            mode="lines+markers",
            name=model,
            line=dict(color=MODEL_COLORS.get(model, "#888"), width=2.5),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a1a2e"), x=0),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, title=METRIC_LABELS[metric]),
        xaxis=dict(showgrid=False, title="Fold", tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=40, r=10),
        height=320,
    )
    return fig


def _box_chart(results_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for model, grp in results_df.groupby("Model"):
        fig.add_trace(go.Box(
            y=grp[metric],
            name=model,
            marker_color=MODEL_COLORS.get(model, "#888"),
            line_color=MODEL_COLORS.get(model, "#888"),
            fillcolor=_hex_to_rgba(MODEL_COLORS.get(model, "#888888"), 0.25),
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker_size=7,
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a1a2e"), x=0),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#ececec", zeroline=False, title=METRIC_LABELS[metric]),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(t=40, b=20, l=40, r=10),
        height=300,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _radar_chart(summary_df: pd.DataFrame) -> go.Figure:
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    labels = [METRIC_LABELS[m] for m in radar_metrics]
    fig = go.Figure()
    for model, grp in summary_df.groupby("Model"):
        vals = [grp.loc[grp["Metric"] == m, "Mean"].values[0] for m in radar_metrics]
        vals += vals[:1]
        color = MODEL_COLORS.get(model, "#888888")
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels + labels[:1],
            fill="toself",
            name=model,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.2),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#ddd")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="#ffffff",
        margin=dict(t=20, b=60, l=40, r=40),
        height=400,
        title=dict(text="Overall Profile (mean across folds)", font=dict(size=15, color="#1a1a2e"), x=0),
    )
    return fig


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
| **Log Loss** | Average log-probability assigned to the true label; heavily penalises confident wrong predictions — lower is better. |
| **Brier Score** | Mean squared error between predicted probabilities and true binary outcomes; lower is better. |
""")

if st.button("Run Model Comparison", type="primary"):

    discretized_dataset = pd.read_csv(
        f"data/dynops_discretized_dataset_{date.today().strftime('%d-%m-%y')}.csv"
    )

    gkf = GroupKFold(n_splits=5)
    results = []
    progress = st.progress(0, text="Running fold 1 / 5…")

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(discretized_dataset, groups=discretized_dataset["Chiffre"])
    ):
        progress.progress((fold) / 5, text=f"Running fold {fold + 1} / 5…")

        train_df = discretized_dataset.iloc[train_idx].copy()
        test_df  = discretized_dataset.iloc[test_idx].copy()
        y_true   = test_df["full_remission"].astype(int)

        # ── DBN ───────────────────────────────────────────────────────────────
        train_dataset, edges = learn_structure(train_df.copy(), target_variable="full_remission")
        network        = create_network_dict(edges, train_dataset.columns)
        net_params     = compute_beta_params(train_dataset, network)
        dbn_prob       = predict_node_probability(test_df, "full_remission", net_params)
        m = evaluate_predictions(y_true, dbn_prob)
        m.update({"Model": "DBN", "Fold": fold + 1})
        results.append(m)

        # ── baselines ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = build_baseline_features(train_df, test_df)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        m = evaluate_predictions(y_test, lr.predict_proba(X_test)[:, 1])
        m.update({"Model": "LogisticRegression", "Fold": fold + 1})
        results.append(m)

        rf = RandomForestClassifier(n_estimators=500, random_state=23, n_jobs=-1)
        rf.fit(X_train, y_train)
        m = evaluate_predictions(y_test, rf.predict_proba(X_test)[:, 1])
        m.update({"Model": "RandomForest", "Fold": fold + 1})
        results.append(m)

        if HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=23,
                eval_metric="logloss",
            )
            xgb.fit(X_train, y_train)
            m = evaluate_predictions(y_test, xgb.predict_proba(X_test)[:, 1])
            m.update({"Model": "XGBoost", "Fold": fold + 1})
            results.append(m)

    progress.progress(1.0, text="Done!")

    # ── aggregate ─────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    summary_long = (
        results_df
        .melt(id_vars=["Model", "Fold"], value_vars=ALL_METRICS, var_name="Metric", value_name="Value")
        .groupby(["Model", "Metric"])
        .agg(Mean=("Value", "mean"), Std=("Value", "std"))
        .reset_index()
    )

    # ── ① ranking cards ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Ranking")
    st.caption("Ranked by mean ROC-AUC across folds. ↑ higher is better  ↓ lower is better")

    ranking = (
        results_df.groupby("Model")
        .agg(
            ROC_AUC=("ROC_AUC", "mean"), PR_AUC=("PR_AUC", "mean"),
            F1=("F1", "mean"), Accuracy=("Accuracy", "mean"),
            LogLoss=("LogLoss", "mean"), Brier=("Brier", "mean"),
        )
        .sort_values("ROC_AUC", ascending=False)
        .reset_index()
    )

    cols = st.columns(len(ranking))
    for i, (_, row) in enumerate(ranking.iterrows()):
        medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
        with cols[i]:
            color = MODEL_COLORS.get(row["Model"], "#888")
            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:12px;
                            padding:16px 12px;text-align:center;background:#fafafa;">
                  <div style="font-size:26px">{medal}</div>
                  <div style="font-weight:700;font-size:15px;color:{color};
                              margin:6px 0 10px">{row['Model']}</div>
                  <div style="font-size:12px;color:#555;line-height:1.9">
                    ROC-AUC <b>{row['ROC_AUC']:.3f}</b><br>
                    PR-AUC &nbsp;<b>{row['PR_AUC']:.3f}</b><br>
                    F1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{row['F1']:.3f}</b><br>
                    Accuracy <b>{row['Accuracy']:.3f}</b><br>
                    Log Loss &nbsp;<b>{row['LogLoss']:.3f}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── ② radar overview ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Overall Profile")
    st.plotly_chart(_radar_chart(summary_long), use_container_width=True)

    # ── ③ grouped bar — all classification metrics ────────────────────────────
    st.divider()
    st.subheader("Classification Metrics — Mean across Folds")
    higher_long = summary_long[summary_long["Metric"].isin(METRICS_HIGHER)]
    st.plotly_chart(
        _grouped_bar(higher_long, METRICS_HIGHER, "Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC"),
        use_container_width=True,
    )

    # Loss metrics side by side
    c1, c2 = st.columns(2)
    for metric, col in [("LogLoss", c1), ("Brier", c2)]:
        sub = summary_long[summary_long["Metric"] == metric]
        col.plotly_chart(
            _bar_chart(sub.rename(columns={}), metric, f"{METRIC_LABELS[metric]} (Mean ± SD) ↓ lower is better"),
            use_container_width=True,
        )

    # ── ④ per-metric bar charts (ROC + PR) ───────────────────────────────────
    st.divider()
    st.subheader("Key Metrics — Mean ± SD")
    c1, c2 = st.columns(2)
    for metric, col in [("ROC_AUC", c1), ("PR_AUC", c2)]:
        sub = summary_long[summary_long["Metric"] == metric]
        col.plotly_chart(
            _bar_chart(sub, metric, f"{METRIC_LABELS[metric]} (Mean ± SD)"),
            use_container_width=True,
        )
    c1, c2 = st.columns(2)
    for metric, col in [("F1", c1), ("Accuracy", c2)]:
        sub = summary_long[summary_long["Metric"] == metric]
        col.plotly_chart(
            _bar_chart(sub, metric, f"{METRIC_LABELS[metric]} (Mean ± SD)"),
            use_container_width=True,
        )

    # ── ⑤ fold-level line charts ──────────────────────────────────────────────
    st.divider()
    st.subheader("Performance Across Folds")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_line_chart(results_df, "ROC_AUC", "ROC-AUC per Fold"), use_container_width=True)
    c2.plotly_chart(_line_chart(results_df, "PR_AUC",  "PR-AUC per Fold"),  use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(_line_chart(results_df, "F1",       "F1 per Fold"),      use_container_width=True)
    c2.plotly_chart(_line_chart(results_df, "Accuracy", "Accuracy per Fold"),use_container_width=True)

    # ── ⑥ box plots ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Distribution Across Folds")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_box_chart(results_df, "ROC_AUC", "ROC-AUC Distribution"), use_container_width=True)
    c2.plotly_chart(_box_chart(results_df, "PR_AUC",  "PR-AUC Distribution"),  use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(_box_chart(results_df, "F1",      "F1 Distribution"),      use_container_width=True)
    c2.plotly_chart(_box_chart(results_df, "Brier",   "Brier Distribution"),   use_container_width=True)

    # ── ⑦ raw data tables ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("Fold-level results table"):
        st.dataframe(
            results_df.style.format({m: "{:.4f}" for m in ALL_METRICS}),
            use_container_width=True,
        )

    with st.expander("Cross-validation summary (mean ± std)"):
        pivot = summary_long.copy()
        pivot["Mean ± Std"] = pivot.apply(lambda r: f"{r['Mean']:.3f} ± {r['Std']:.3f}", axis=1)
        tbl = pivot.pivot(index="Model", columns="Metric", values="Mean ± Std")[ALL_METRICS]
        tbl.columns = [METRIC_LABELS[m] for m in tbl.columns]
        st.dataframe(tbl, use_container_width=True)

if st.button("Back: Evaluation", type="primary"):
    st.switch_page("pages/Evaluation.py")