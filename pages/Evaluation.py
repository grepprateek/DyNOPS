import streamlit as st
import pandas as pd
from datetime import date
import itertools
import json
import numpy as np
import re
from collections import defaultdict
from typing import List

from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, PC, GES
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

st.set_page_config(layout="wide")

st.html("""
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
""")

st.header("Evaluation")

# ── helpers ───────────────────────────────────────────────────────────────────

def define_forbidden_edges(dataset: pd.DataFrame, target_variable: str = None) -> List:
    forbidden_edges = []
    if target_variable:
        forbidden_edges.extend(
            [(target_variable, var) for var in dataset.columns if var != target_variable]
        )
    forbidden_edges.extend(
        (var1, var2)
        for var1, var2 in itertools.permutations(dataset.columns, 2)
        if var1.endswith("(t)") and var2.endswith("(tm1)")
    )
    forbidden_edges.extend(
        (var1, var2)
        for var1, var2 in itertools.permutations(dataset.columns, 2)
        if ("(tm" in var1 and "(tm" in var2)
        and (int(var1.split("_")[1][3]) < int(var2.split("_")[1][3]))
    )
    return forbidden_edges


def learn_structure(
    dataset: pd.DataFrame,
    algorithm: str = "hcs",
    target_variable: str = None,
) -> List:
    dataset = dataset.drop(columns=["Chiffre", "Date"])
    for col in dataset.columns:
        dataset[col] = dataset[col].astype("category")
    forbidden_edges = define_forbidden_edges(dataset, target_variable)
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
    if algorithm == "hcs":
        model = HillClimbSearch(data=dataset).estimate(
            scoring_method="bic-d", expert_knowledge=expert_knowledge
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return dataset, model.edges


def get_timepoint(node: str):
    match = re.search(r"\((t|tm\d+)\)", node)
    return match.group(1) if match else None


def add_delay_edges(edges, dataset_with_delay: pd.DataFrame):
    delay_cols = {c for c in dataset_with_delay.columns if c.startswith("delay_")}
    augmented = list(edges)
    for parent, child in edges:
        p_tp = get_timepoint(parent)
        c_tp = get_timepoint(child)
        if p_tp is None or c_tp is None or p_tp == c_tp:
            continue
        delay_name = f"delay_{p_tp}_{c_tp}"
        if delay_name not in delay_cols:
            delay_name = f"delay_{c_tp}_{p_tp}"
        if delay_name in delay_cols:
            augmented.append((delay_name, child))
    return augmented


def create_network_dict(edges, dataset_columns) -> dict:
    parents = defaultdict(list)
    for parent, child in edges:
        parents[child].append(parent)
    return {node: list(parents[node]) for node in dataset_columns}


# ── prediction from JSON params ───────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_node_probability(
    test_df: pd.DataFrame,
    node: str,
    network_params: dict,
) -> np.ndarray:
    """
    Return P(node == 1 | parents) for every row in test_df.

    Parameters are loaded from JSON — they store raw logistic regression
    coefficients in the layout:
        params[k] = [intercept, coef_1, coef_2, ...]   for class k
    (one row per non-reference class; binary nodes have exactly 1 row).
    """
    info = network_params[node]

    if not info["has_parents"]:
        # Marginal probability of class "1"
        p1 = float(info["params"].get("1", info["params"].get(1, 0.0)))
        return np.full(len(test_df), p1)

    parents = info["parents"]
    coef_matrix = np.array(info["params"])   # shape: (n_classes-1, 1+n_features)

    # Reconstruct get_dummies(drop_first=True) feature matrix
    X_source = test_df[parents].copy()
    X_dummies = pd.get_dummies(X_source, drop_first=True, dtype=int)

    # Align columns to whatever was seen at training time
    # (the JSON doesn't store feature names, so we rely on same column order)
    X = X_dummies.values.astype(float)

    # Prepend intercept column
    intercept = np.ones((X.shape[0], 1))
    X_full = np.hstack([intercept, X])   # (n_samples, 1+n_features)

    # logits for each non-reference class: (n_samples, n_classes-1)
    logits_pos = X_full @ coef_matrix.T

    if coef_matrix.shape[0] == 1:
        # Binary: single logit → sigmoid
        prob_class1 = 1.0 / (1.0 + np.exp(-logits_pos[:, 0]))
        return prob_class1
    else:
        # Multiclass: prepend zero logit for reference class, then softmax
        logits_ref = np.zeros((X_full.shape[0], 1))
        logits_all = np.hstack([logits_ref, logits_pos])
        probs = _softmax(logits_all)
        return probs[:, 1]   # P(class index 1)


# ── UI ────────────────────────────────────────────────────────────────────────

PARAMS_PATH = "network_parameters.json"

if st.button("Get evaluation results", type="primary"):

    # ── load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        df_no_delay = pd.read_csv(
            f"data/dynops_discretized_dataset_without_delay_{date.today().strftime('%d-%m-%y')}.csv"
        )
        df_with_delay = pd.read_csv(
            f"data/dynops_discretized_dataset_with_delay_{date.today().strftime('%d-%m-%y')}.csv"
        )

    # ── train / test split (group by patient) ─────────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
    train_idx, test_idx = next(
        splitter.split(df_no_delay, groups=df_no_delay["Chiffre"])
    )

    train_no_delay  = df_no_delay.iloc[train_idx]
    test_no_delay   = df_no_delay.iloc[test_idx]
    train_with_delay = df_with_delay.iloc[train_idx]
    test_with_delay  = df_with_delay.iloc[test_idx]

    # ── load network parameters from JSON ─────────────────────────────────────
    with st.spinner("Loading network parameters…"):
        try:
            with open(PARAMS_PATH) as f:
                network_params = json.load(f)
   
        except FileNotFoundError:
            st.warning(
                f"`{PARAMS_PATH}` not found — re-learning structure and parameters from training data."
            )
            structure_dataset, edges = learn_structure(
                train_no_delay.copy(), algorithm="hcs", target_variable="full_remission"
            )
            edges = add_delay_edges(edges, train_with_delay)
            parameter_dataset = train_with_delay.drop(columns=["Chiffre", "Date"])
            network_dict = create_network_dict(edges, parameter_dataset.columns)

            from sklearn.linear_model import LogisticRegression

            def _compute_params(dataset, network_edges):
                out = {}
                for var in dataset.columns:
                    pars = network_edges[var]
                    if not pars:
                        out[var] = {
                            "has_parents": False,
                            "params": dataset[var]
                            .value_counts(normalize=True)
                            .sort_index()
                            .to_dict(),
                        }
                    else:
                        X = pd.get_dummies(dataset[pars], drop_first=True, dtype=int)
                        y = pd.Categorical(dataset[var]).codes
                        lr = LogisticRegression(
                            solver="lbfgs", fit_intercept=True, max_iter=500
                        )
                        lr.fit(X, y)
                        intercepts = lr.intercept_.reshape(-1, 1)
                        coef_with_intercept = np.hstack([intercepts, lr.coef_])
                        out[var] = {
                            "has_parents": True,
                            "parents": pars,
                            "params": coef_with_intercept.tolist(),
                        }
                return out

            network_params = _compute_params(parameter_dataset, network_dict)

    # ── predictions ───────────────────────────────────────────────────────────
    with st.spinner("Computing predictions…"):
        y_true = test_with_delay["full_remission"].astype(int).values
        y_prob = predict_node_probability(test_with_delay, "full_remission", network_params)
        y_pred = (y_prob >= 0.5).astype(int)

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
        "PR-AUC":    average_precision_score(y_true, y_prob),
        "Log Loss":  log_loss(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }

    # ── render ────────────────────────────────────────────────────────────────
    st.subheader("DBN Performance")

    # Metric cards
    col_pairs = [
        ("Accuracy",  "Precision"),
        ("Recall",    "F1"),
        ("ROC-AUC",   "PR-AUC"),
        ("Log Loss",  "Brier"),
    ]

    for left, right in col_pairs:
        c1, c2 = st.columns(2)
        lv, rv = metrics[left], metrics[right]
        c1.metric(label=left,  value=f"{lv:.4f}")
        c2.metric(label=right, value=f"{rv:.4f}")

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"],
    )
    st.dataframe(cm_df, use_container_width=False)


if st.button("← Back: Structure Learning", type="primary"):
    st.switch_page("pages/Structure_Learning.py")

if st.button("Next: Baselines Comparison", type="primary"):
    st.switch_page("pages/Baselines_Comparison.py")