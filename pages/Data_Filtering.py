import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import inspect

def filter_wrt_year(
    dataset: pd.DataFrame = None,
    start_year: int = 2018,
    end_year: int = date.today().year):
    """
    Filters dataset by selecting a subset between the start year and the end year.
    """
    dataset['phqdat'] = pd.to_datetime(dataset['phqdat'], errors='coerce')
    dataset['gaddat'] = pd.to_datetime(dataset['gaddat'], errors='coerce')
    dataset = dataset[
        ((dataset['phqdat'].dt.year > start_year) & (dataset['phqdat'].dt.year <= end_year)) &
        ((dataset['gaddat'].dt.year > start_year) & (dataset['gaddat'].dt.year <= end_year)) 
    ]

    dataset = dataset.reset_index(drop=True)

    return dataset
    
def filter_wrt_instance(
    dataset: pd.DataFrame = None,
    only_therapy_sessions: bool = True
):
    """
    Filters dataset that contains only the therapy sessions conducted at PTA and not the ones before or after.
    """
    if only_therapy_sessions:
        dataset = dataset[dataset['instance_phq'].str.startswith('T', na = False) &
                          dataset['instance_gad'].str.startswith('T', na = False)]
        
        dataset = dataset.reset_index(drop=True)
        return dataset
    
def filter_wrt_num_sessions(
    dataset: pd.DataFrame = None,
    max_sessions: int = None):
    """
    Filters dataset for a maximum number of required sessions
    """
    dataset = dataset[dataset['session_num'] <= max_sessions]
    dataset = dataset.drop(columns = ['session_num'])
    dataset = dataset.reset_index(drop=True)
    return dataset

def filter_wrt_num_questionnaire_sets(
    dataset: pd.DataFrame = None,
    max_questionnaire_sets: int = None):
    """
    Filters dataset for a maximum number of PHQ/GAD questionnaire sets
    """
    phq_cols = dataset.filter(regex=r'^phq\d+$').columns
    phq_cols_to_remove = [col for col in phq_cols if int(col[3:]) > 5]
    phq_cols_to_keep = [col for col in phq_cols if col not in phq_cols_to_remove]
    print(phq_cols_to_remove)

    gad_cols = dataset.filter(regex=r'^gad\d+$').columns
    gad_cols_to_remove = [col for col in gad_cols if int(col[3:]) > 5]
    gad_cols_to_keep = [col for col in gad_cols if col not in gad_cols_to_remove]
    print(gad_cols_to_remove)

    dataset = dataset.drop(columns = phq_cols_to_remove + gad_cols_to_remove + ['phqsum', 'gadsum'])
    
    dataset['phqsum'] = dataset[phq_cols_to_keep].sum(axis=1)
    dataset['gadsum'] = dataset[gad_cols_to_keep].sum(axis=1)

    dataset = dataset.reset_index(drop=True)
    return dataset

# USER INTERFACE

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

st.header("Data Filtering")

long_format_dataset = pd.read_csv(f"data/dynops_long_format_dataset_{date.today().strftime('%d-%m-%y')}.csv")

# 1. Filter with respect to year 

long_format_dataset['phqdat'] = pd.to_datetime(long_format_dataset['phqdat'], errors='coerce')
long_format_dataset['gaddat'] = pd.to_datetime(long_format_dataset['gaddat'], errors='coerce')
min_year = int(long_format_dataset['phqdat'].dt.year.min())
max_year = int(long_format_dataset['phqdat'].dt.year.max())
year_options = np.arange(min_year, max_year+1)

st.write("### 1. Filter with respect to session year.")
with st.expander("View source code"):
        code = inspect.getsource(filter_wrt_year)
        st.code(code, language='python')

col1, col2 = st.columns([1, 1])
with col1:
    start_year = st.selectbox(label = "Start Year", options = year_options)
    
with col2:
    end_year = st.selectbox(label = "End Year", options = year_options, index=len(year_options)-1)

# 2. Filter with respect to session instance 

st.write("### 2. Filter with respect to therapy sessions conducted.")
st.write("Only sessions with instance starting with 'T' selected")
with st.expander("View source code"):
        code = inspect.getsource(filter_wrt_instance)
        st.code(code, language='python')

only_therapy_sessions = st.checkbox("Only therapy sessions")
all_therapy_sessions = st.checkbox("All sessions")
dataset_filtered_wrt_instance = True if only_therapy_sessions else False

# 3. Filter with respect to number of sessions

st.write("### 3. Filter with respect to number of therapy sessions")
st.write("Select the maximum number of required sessions per patient")
with st.expander("View source code"):
    code = inspect.getsource(filter_wrt_num_sessions)
    st.code(code, language='python')

max_sessions = st.number_input("Enter the number of required therapy sessions", min_value=1, step=1, value=24)

# 4. Filter with respect to number of questionnaire sets
st.write("### 4. Filter with respect to number of questionnaire sets")
st.write("Select the maximum number of required questionnaire sets per patient")
with st.expander("View source code"):
    code = inspect.getsource(filter_wrt_num_questionnaire_sets)
    st.code(code, language='python')

max_questionnaire_sets = st.number_input("Enter the number of required questionnaire sets", max_value=5, step=1, value=5)

st.write("Click the following button to filter the datset based on the selected options.")
if st.button("Filter Dataset", type='primary'):
    dataset_filtered_wrt_year = filter_wrt_year(long_format_dataset, start_year, end_year)
    dataset_filtered_wrt_instance = filter_wrt_instance(dataset_filtered_wrt_year, only_therapy_sessions)
    dataset_filtered_wrt_num_sessions = filter_wrt_num_sessions(dataset_filtered_wrt_instance, max_sessions)
    dataset_filtered_wrt_num_questionnaires = filter_wrt_num_questionnaire_sets(dataset_filtered_wrt_num_sessions, max_questionnaire_sets)
    st.dataframe(dataset_filtered_wrt_num_questionnaires)
    dataset_filtered_wrt_num_questionnaires.to_csv(f"data/dynops_filtered_dataset_{date.today().strftime('%d-%m-%y')}.csv", index=False)
    st.write(f"The filtered dataset has {dataset_filtered_wrt_num_questionnaires.shape[0]} rows and {dataset_filtered_wrt_num_questionnaires.shape[1]} columns.")

if st.button("Next: Data Cleaning", type='primary'):
    st.switch_page("pages/Data_Cleaning.py")
