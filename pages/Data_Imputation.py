import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

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

st.header("Data Imputation")

st.text("In this step, missing values in the cleaned dataset are imputed based on variable types as follows:")
st.markdown("""
    1. Categorical variables: Mode
    2. Continuous variables: Median (skewed data distribution)
    3. Time-dependent (questionnaire) variables: Forward Fill followed by Backward Fill
""")


if st.button("Impute Dataset", type='primary'):
    cleaned_dataset = pd.read_csv(f"data/dynops_cleaned_dataset_{date.today().strftime('%d-%m-%y')}.csv")
    imputed_dataset = cleaned_dataset.copy()

    # variables: BDIIISummenwertPrae and BDIIISummenwertPost
    imputed_dataset['BDIIISummenwertPrae'] = imputed_dataset['BDIIISummenwertPrae'].fillna(imputed_dataset['BDIIISummenwertPrae'].median()).astype(int)
    imputed_dataset['BDIIISummenwertPost'] = imputed_dataset['BDIIISummenwertPost'].fillna(imputed_dataset['BDIIISummenwertPost'].median()).astype(int)
    
    # variable: Beruf
    imputed_dataset['Beruf'] = imputed_dataset['Beruf'].fillna(imputed_dataset['Beruf'].mode()[0])

    # variable: Beziehung
    imputed_dataset['Beziehung'] = imputed_dataset['Beziehung'].fillna(imputed_dataset['Beziehung'].mode()[0])

    # variable: Qualifikation
    imputed_dataset['Qualifikation'] = imputed_dataset['Qualifikation'].fillna(imputed_dataset['Qualifikation'].mode()[0])

    # variable: Schulabschluss
    imputed_dataset['Schulabschluss'] = imputed_dataset['Schulabschluss'].fillna(imputed_dataset['Schulabschluss'].mode()[0])

    # variables: SCL90_GSI_Prae and SCL90_GSI_Post
    imputed_dataset['SCL90_GSI_Prae'] = imputed_dataset['SCL90_GSI_Prae'].fillna(imputed_dataset['SCL90_GSI_Prae'].median())
    imputed_dataset['SCL90_GSI_Post'] = imputed_dataset['SCL90_GSI_Post'].fillna(imputed_dataset['SCL90_GSI_Post'].median())
    
    # variable: PrimDiag_Kategorie
    imputed_dataset['PrimDiag_Kategorie'] = imputed_dataset['PrimDiag_Kategorie'].fillna(imputed_dataset['PrimDiag_Kategorie'].mode()[0]).astype(int)
    
    # questionnaire variables
    for col in imputed_dataset.columns:
        if col.startswith(('phq', 'gad')) and col not in ['phqsum', 'gadsum']:
            imputed_dataset[col] = imputed_dataset.groupby(by='Chiffre')[col].ffill()
            imputed_dataset[col] = imputed_dataset.groupby(by='Chiffre')[col].bfill()

    # phqsum and gadsum
    phq_cols = [col for col in imputed_dataset.columns if col.startswith('phq') and col != 'phqsum'] 
    gad_cols = [col for col in imputed_dataset.columns if col.startswith('gad') and col != 'gadsum'] 

    imputed_dataset['phqsum'] = imputed_dataset[phq_cols].sum(axis = 1)
    imputed_dataset['gadsum'] = imputed_dataset[gad_cols].sum(axis = 1)

    imputed_dataset.to_csv(f"data/dynops_imputed_dataset_{date.today().strftime('%d-%m-%y')}.csv", index=False)
    st.dataframe(imputed_dataset)

if st.button("Imputed Data Visualization", type='primary'):
    st.switch_page("pages/Imputed_Data_Visualization.py")

if st.button("Next: Data Discretization", type='primary'):
    st.switch_page("pages/Data_Discretization.py")