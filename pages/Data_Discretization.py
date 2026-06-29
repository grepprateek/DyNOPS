import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from itertools import combinations

def create_lagged_dataset(discretized_dataset: pd.DataFrame, max_lag: int):
    for qc in questionnaire_cols:
        discretized_dataset[qc] = discretized_dataset[qc].astype(int)
        discretized_dataset[f"{qc}_(t)"] = discretized_dataset[qc]
        for lag in range(1, max_lag+1):
            column_name = f"{qc}_(tm{lag})"
            discretized_dataset[column_name] = (discretized_dataset.groupby('Chiffre')[f"{qc}_(t)"].shift(lag).astype('Int64'))

    lag_cols = [f"{qc}_(tm{lag})" for qc in questionnaire_cols for lag in range(1, max_lag + 1)]
    discretized_dataset = discretized_dataset.dropna(subset=lag_cols).copy()
    discretized_dataset = discretized_dataset.drop(columns=questionnaire_cols)
    return discretized_dataset


def create_delay_lagged_dataset(discretized_dataset: pd.DataFrame, questionnaire_cols: list, max_lag: int):

    df = discretized_dataset.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Create current and lagged questionnaire variables
    for qc in questionnaire_cols:
        df[qc] = df[qc].astype(int)
        df[f"{qc}_(t)"] = df[qc]

        for lag in range(1, max_lag + 1):
            df[f"{qc}_(tm{lag})"] = df.groupby("Chiffre")[qc].shift(lag).astype("Int64")

    # Create lagged dates
    for lag in range(1, max_lag + 1):
        df[f"Date_tm{lag}"] = df.groupby("Chiffre")["Date"].shift(lag)

    # Create all pairwise time differences
    timepoints = ["t"] + [f"tm{i}" for i in range(1, max_lag + 1)]

    for tp1, tp2 in combinations(timepoints, 2):
        col1 = "Date" if tp1 == "t" else f"Date_{tp1}"
        col2 = "Date" if tp2 == "t" else f"Date_{tp2}"

        df[f"delay_{tp2}_{tp1}"] = (df[col1] - df[col2]).dt.days.abs()

    # Remove rows without complete lag history
    lag_cols = [f"{qc}_(tm{lag})" for qc in questionnaire_cols for lag in range(1, max_lag + 1)]
    delay_cols = [col for col in df.columns if col.startswith("delay_")]

    df = df.dropna(subset=lag_cols + delay_cols).copy()

    # Categorize delays
    for col in delay_cols:
        df[col] = pd.cut(
            df[col],
            bins=[-1, 7, 30, 90, 365, float("inf")],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

    # Remove original questionnaire columns and helper date columns
    df = df.drop(columns=questionnaire_cols)
    df = df.drop(columns=[f"Date_tm{lag}" for lag in range(1, max_lag + 1)])

    return df

st.html(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #3b766c; /* Dark Green background */
            color: white;              /* Text color */
            border-color: black;      /* Border color */
        }
        
        /* Optional: Change color when hovering over the button */
        div.stButton > button[kind="primary"]:hover {
            background-color: #1B5E20;
            color: white;
        }
        </style>
        """
    )

st.header("Data Discretization")

col1, col2 = st.columns([4, 1], vertical_alignment='center')

with col1:
    lag = st.number_input("Enter the required model memory:", min_value=1, step=1)

with col2:
    st.markdown(
        "<div style='padding-top: 28px;'></div>", unsafe_allow_html=True
    )
    discretize_button = st.button("Discretize Data", type='primary')

if discretize_button:
    imputed_data = pd.read_csv(f"data/dynops_imputed_dataset_{date.today().strftime('%d-%m-%y')}.csv")
    discretized_data = imputed_data.copy()

    continuous_variables = ['age', 'BDIIISummenwertPrae', 'BDIIISummenwertPost', 'SCL90_GSI_Prae', 'SCL90_GSI_Post', 'phqsum', 'gadsum']
    for cv in continuous_variables:
        discretized_data[cv] = pd.qcut(discretized_data[cv], q=4, labels = [0, 1, 2, 3])
    
    categorical_columns = ['Geschlecht_PP', 'Beruf', 'Beziehung', 'Qualifikation', 'Schulabschluss']
    for cc in categorical_columns:
        discretized_data[cc] = discretized_data[cc].astype('category').cat.codes
    
    questionnaire_cols = ['phq001', 'phq002', 'phq003', 'phq004', 'phq005', 'gad001', 'gad002', 'gad003', 'gad004', 'gad005', 'phqsum', 'gadsum']
    
    discretized_data_without_delay = create_lagged_dataset(discretized_data, lag)
    discretized_data_with_delay = create_delay_lagged_dataset(discretized_data, questionnaire_cols, lag)
    st.dataframe(discretized_data_without_delay)
    discretized_data_without_delay.to_csv(f"data/dynops_discretized_dataset_without_delay_{date.today().strftime('%d-%m-%y')}.csv", index = False)
    discretized_data_with_delay.to_csv(f"data/dynops_discretized_dataset_with_delay_{date.today().strftime('%d-%m-%y')}.csv", index = False)

if st.button("Next: Structure Learning", type='primary'):
    st.switch_page("pages/Structure_Learning.py")