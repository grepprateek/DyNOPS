import streamlit as st
import inspect
import numpy as np
import pandas as pd
from datetime import date
from collections import Counter
import json
import matplotlib.pyplot as plt
import plotly.express as px

def patient_id_eda(dataset: pd.DataFrame, 
                   id_column: str = 'Chiffre'):
    """
    Returns the unique patient IDs and the total number of patients.

    Attributes
    ----------
    dataset: pd.DataFrame
        last-saved long format dataset fetched from the `data` directory
    id_column: str
        Patient ID column
    
    Returns
    -------
    First 10 unique IDs, total number of patients
    """
    ids_sequence = dataset[id_column]
    unique_ids = ids_sequence.unique()
    total_patients = len(unique_ids)
    return unique_ids[:10], total_patients

def sessions_data_eda(dataset: pd.DataFrame,
                    id_column: str = 'Chiffre',
                    sessions_column: str = None,
                    num_min_sessions: int = 10):
    
    """
    Returns the number of patients who attended therapy sessions less than a certain threshold.

    Attributes
    ----------
    dataset: pd.DataFrame
        last-saved long format dataset fetched from the `data` directory
    id_column: str
        [patient ID column
    sessions_column: 
        therapy session number column
    num_min_sessions:
        minimum threshold value for attended therapy sessions
    """

    sessions_data = dataset[[id_column, sessions_column]].drop_duplicates()
    sessions_data_counts = (
        sessions_data[sessions_column]
        .value_counts()
        .sort_index()
    )
    total_missing_values = sessions_data[
        sessions_data[sessions_column].isna()
    ][id_column].nunique()

    # Patients having fewer sessions than threshold
    patients_less_sessions = sessions_data[sessions_data[sessions_column] < num_min_sessions][id_column].unique()
    num_patients_less_sessions = len(patients_less_sessions)

    return (sessions_data_counts, num_patients_less_sessions, total_missing_values)

def bdi_prae_data_eda(dataset: pd.DataFrame):
    bdiii_prae_data = dataset['BDIIISummenwertPrae'].unique()
    bdiii_prae_data = bdiii_prae_data[~np.isnan(np.array(bdiii_prae_data))]
    mean_bdiii_prae = bdiii_prae_data.mean()
    outliers = bdiii_prae_data[bdiii_prae_data < 0]
    return bdiii_prae_data, outliers, mean_bdiii_prae 

def bdi_post_data_eda(dataset: pd.DataFrame):
    bdiii_post_data = dataset['BDIIISummenwertPost'].unique()
    bdiii_post_data = bdiii_post_data[~np.isnan(np.array(bdiii_post_data))]
    mean_bdiii_post = bdiii_post_data.mean()
    outliers = bdiii_post_data[bdiii_post_data < 0]
    return bdiii_post_data, outliers, mean_bdiii_post 

def beruf_data_eda(dataset: pd.DataFrame):
    beruf_data = dataset[["Chiffre", "Beruf"]].drop_duplicates()
    return beruf_data, beruf_data['Beruf']

def beziehung_data_eda(dataset: pd.DataFrame):
    beziehung_data = dataset[["Chiffre", "Beziehung"]].drop_duplicates()
    missing_values = beziehung_data["Beziehung"].isna().sum()
    beziehung_data = beziehung_data.dropna()
    return beziehung_data, beziehung_data['Beziehung'], missing_values

def diag_count_data_eda(dataset: pd.DataFrame):
    diag_count_data = dataset[["Chiffre", "Diag_Count"]].drop_duplicates()
    return diag_count_data, diag_count_data['Diag_Count']

def geschlecht_pp_data_eda(dataset: pd.DataFrame):
    geschlecht_pp_data = dataset[["Chiffre", "Geschlecht_PP"]].drop_duplicates()
    return geschlecht_pp_data, geschlecht_pp_data['Geschlecht_PP']

def primdiag_kategorie_data_eda(dataset: pd.DataFrame):
    primdiag_kategorie_data = dataset[["Chiffre", "PrimDiag_Kategorie"]].drop_duplicates()
    missing_values = primdiag_kategorie_data["PrimDiag_Kategorie"].isna().sum()
    primdiag_kategorie_data = primdiag_kategorie_data.dropna()
    return primdiag_kategorie_data, primdiag_kategorie_data['PrimDiag_Kategorie'], missing_values

def qualifikation_data_eda(dataset: pd.DataFrame):
    qualifikation_data = dataset[["Chiffre", "Qualifikation"]].drop_duplicates()
    missing_values = qualifikation_data["Qualifikation"].isna().sum()
    qualifikation_data = qualifikation_data.dropna()
    return qualifikation_data, qualifikation_data['Qualifikation'], missing_values

def scl90_gsi_prae_data_eda(dataset: pd.DataFrame):
    dataset = dataset[["Chiffre", "SCL90_GSI_Prae"]].drop_duplicates()
    missing_values = dataset["SCL90_GSI_Prae"].isna().sum()
    dataset['SCL90_GSI_Prae'] = dataset['SCL90_GSI_Prae'].str.replace(',', '.', regex=False).astype(float).round(2)
    dataset = dataset.dropna()
    return dataset, missing_values

def scl90_gsi_post_data_eda(dataset: pd.DataFrame):
    dataset = dataset[["Chiffre", "SCL90_GSI_Post"]].drop_duplicates()
    missing_values = dataset["SCL90_GSI_Post"].isna().sum()
    dataset['SCL90_GSI_Post'] = dataset['SCL90_GSI_Post'].str.replace(',', '.', regex=False).astype(float).round(2)
    dataset = dataset.dropna()
    return dataset, missing_values

def schulabschluss_data_eda(dataset: pd.DataFrame):
    schulabschluss_data = dataset[["Chiffre", "Schulabschluss"]].drop_duplicates()
    missing_values = schulabschluss_data["Schulabschluss"].isna().sum()
    schulabschluss_data = schulabschluss_data.dropna()
    return schulabschluss_data, schulabschluss_data['Schulabschluss'], missing_values

def therapie_status_data_eda(dataset: pd.DataFrame):
    therapie_status_data = dataset[["Chiffre", "TherapieStatus_PP"]].drop_duplicates()
    missing_values = therapie_status_data["TherapieStatus_PP"].isna().sum()
    therapie_status_data = therapie_status_data.dropna()
    return therapie_status_data, therapie_status_data['TherapieStatus_PP'], missing_values

def age_data_eda(dataset: pd.DataFrame):
    age_data = dataset[["Chiffre", "age"]]
    missing_values = age_data["age"].isna().sum()
    age_data = age_data['age'].astype(float).dropna().values
    q1 = np.percentile(age_data, 25)
    q3 = np.percentile(age_data, 75)
    iqr = q3 - q1

    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    # fliers = np.where((age_data <= 0) | (age_data > upper_whisker), age_data, np.nan)    
    outliers = age_data[age_data <= 0]
    return dataset, age_data, outliers, missing_values

def full_remission_data_eda(dataset: pd.DataFrame):
    full_remission_data = dataset[["Chiffre", "full_remission"]].drop_duplicates()
    missing_values = full_remission_data["full_remission"].isna().sum()
    full_remission_data = full_remission_data.dropna()
    return full_remission_data, full_remission_data['full_remission'], missing_values

def time_series_data_eda(dataset: pd.DataFrame, id_column: str = 'Chiffre', date_column = 'phqdat', patient_id: str = None):
    dataset = dataset.sort_values(by=[id_column, date_column])
    if patient_id:
        dataset = dataset[dataset[id_column] == patient_id]
    return dataset

# Web-app

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

st.header("Exploratory Data Analysis")
col1, col2 = st.columns([1,1])

with col1:
    long_format_dataset = pd.read_csv(f"data/dynops_long_format_dataset_{date.today().strftime('%d-%m-%y')}.csv")
    columns = long_format_dataset.columns.to_list()
    cols_to_remove = ['session_num', 'phqdat', 'gaddat', 'instance_phq', 'instance_gad']
    columns = [col for col in columns if col not in cols_to_remove]

with open("pages/metadata.json", "r") as metadata_file:
    metadata = json.load(metadata_file)

selected_variable = st.selectbox(label="Variable", options=columns)
sum_variables = ['phqsum', 'gadsum']
if selected_variable == 'Chiffre':
    st.subheader(f"Distribution of {selected_variable}:")

    with st.expander("View source code"):
        code = inspect.getsource(patient_id_eda)
        st.code(code, language='python')
    
    example_ids, total_patients = patient_id_eda(long_format_dataset)
    
    st.markdown(
        """
        <h5>Unique IDs (first 10)</h5>
        """,
        unsafe_allow_html=True
    )
    st.write(example_ids)
    st.metric(label="Total Patients", value=total_patients)

elif selected_variable in ["Anzahl_BE_Post_PEQ_32", "phqsessions", "gadsessions"]:
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `sessions_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(sessions_data_eda)
        st.code(code, language='python')

    st.subheader(f"Distribution of {selected_variable}")
    
    num_min_sessions = st.slider(
        "Minimum therapy sessions threshold",
        min_value=0,
        max_value=int(
            long_format_dataset[selected_variable].max()
        ),
        value=5,
        step=1
    )

    sessions_data_counts, num_patients_less_sessions, total_missing_values = sessions_data_eda(
        dataset=long_format_dataset,
        sessions_column=selected_variable,
        num_min_sessions=num_min_sessions
    )

    plot_df = (sessions_data_counts.reset_index())

    plot_df.columns = ["Sessions", "Patients"]

    fig = px.bar(
        plot_df,
        x="Sessions",
        y="Patients",
        labels={
            "Sessions": "Therapy Sessions",
            "Patients": "Number of Patients"
        },
        color_discrete_sequence = ['#3b766c']
    )

    fig.add_vline(
        x=num_min_sessions,
        line_dash="dash",
        annotation_text=f"Threshold = {num_min_sessions}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        label=f"Patients with fewer than {num_min_sessions} therapy sessions",
        value=num_patients_less_sessions
    )

    st.metric(
        label=f"Total missing values:",
        value=total_missing_values
    )

elif selected_variable == "BDIIISummenwertPrae":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `bdi_prae_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(bdi_prae_data_eda)
        st.code(code, language='python')

    st.subheader(f"Distribution of {selected_variable}")
    bdiii_data, outliers, mean_bdiii = bdi_prae_data_eda(long_format_dataset)
    fig = px.box(long_format_dataset, y=bdiii_data, title="BDI-II Prae Sum Score Boxplot", color_discrete_sequence=['#3b766c'])
    st.metric(
        label="Average BDI-II Prae sum:",
        value = round(mean_bdiii, 2)
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "BDIIISummenwertPost":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `bdi_post_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(bdi_post_data_eda)
        st.code(code, language='python')

    st.subheader(f"Distribution of {selected_variable}")
    bdiii_data, outliers, mean_bdiii = bdi_post_data_eda(long_format_dataset)
    fig = px.box(long_format_dataset, y=bdiii_data, title="BDI-II Post Sum Score Boxplot", color_discrete_sequence=['#3b766c'])
    st.metric(
        label="Average BDI-II Post sum:",
        value = round(mean_bdiii, 2)
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "Beruf":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `beruf_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(beruf_data_eda)
        st.code(code, language='python')
    st.subheader(f"Distribution of {selected_variable}")
    beruf_data, beruf_col = beruf_data_eda(long_format_dataset)
    fig = px.histogram(beruf_data, x = 'Beruf', color='Beruf', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "Beziehung":
    st.text(metadata[selected_variable]["description"])
    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `beziehung_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(beziehung_data_eda)
        st.code(code, language='python')
    st.subheader(f"Distribution of {selected_variable}")
    beziehung_data, beziehung_col, missing_values = beziehung_data_eda(long_format_dataset)

    fig = px.pie(
        beziehung_data, 
        names='Beziehung', 
        color='Beziehung', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        textinfo='percent+label', 
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    fig.update_layout(showlegend=False)

    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "Diag_Count":
    st.text(metadata[selected_variable]["description"])
    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `diag_count_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(diag_count_data_eda)
        st.code(code, language='python')
        
    st.subheader(f"Distribution of {selected_variable}")
    diag_count_data, diag_count_col = diag_count_data_eda(long_format_dataset)
    sorted_labels = sorted(
        diag_count_data['Diag_Count'].dropna().unique(), 
        key=lambda x: int(x) if str(x).isdigit() else x
    )
    
    fig = px.histogram(
        diag_count_data, 
        x='Diag_Count',
        color='Diag_Count', 
        color_discrete_sequence=px.colors.qualitative.Pastel,
        category_orders={"Diag_Count": sorted_labels}
    )
    
    fig.update_xaxes(type='category', categoryorder='category ascending')
    fig.update_yaxes(title_text="Number of patients")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "Geschlecht_PP":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `geschlecht_pp_data_eda()`
                 Takes parameters ...
            """)
            
    with st.expander("View source code"):
        code = inspect.getsource(geschlecht_pp_data_eda)
        st.code(code, language='python')
        
    st.subheader(f"Distribution of {selected_variable}")
    geschlecht_pp_data, geschlecht_pp_col = geschlecht_pp_data_eda(long_format_dataset)
    geschlecht_pp_data["Geschlecht_PP"] = geschlecht_pp_data["Geschlecht_PP"].map({'w': 'Weiblich', 'm': 'Mannlich'})
    fig = px.pie(
        geschlecht_pp_data, 
        names='Geschlecht_PP', 
        color='Geschlecht_PP', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "PrimDiag_Kategorie":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `primdiag_kategorie_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(primdiag_kategorie_data_eda)
        st.code(code, language='python')
        
    st.subheader(f"Distribution of {selected_variable}")
    primdiag_kategorie_data, primdiag_kategorie_col, missing_values = primdiag_kategorie_data_eda(long_format_dataset)
    
    fig = px.histogram(
        primdiag_kategorie_data, 
        x='PrimDiag_Kategorie', 
        color='PrimDiag_Kategorie', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        hovertemplate="<b>Diag Count:</b> %{x}<br><b>Number of patients:</b> %{y}<extra></extra>"
    )
    
    fig.update_xaxes(
        type='category', 
        categoryorder='category ascending'
    )

    fig.update_yaxes(title_text="Number of patients")
    fig.update_layout(showlegend=False)
    st.metric(
        label="Missing values:",
        value=missing_values
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "Qualifikation":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `qualifikation_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(qualifikation_data_eda)
        st.code(code, language='python')
        
    st.subheader(f"Distribution of {selected_variable}")
    qualifikation_data, qualifikation_col, missing_values = qualifikation_data_eda(long_format_dataset)
    
    fig = px.pie(
        qualifikation_data, 
        names='Qualifikation', 
        color='Qualifikation', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    
    fig.update_layout(showlegend=False)
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "SCL90_GSI_Prae":
    st.text(metadata[selected_variable]["description"])
    import plotly.figure_factory as ff
    
    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `scl90_gsi_prae_data_eda()`
                 Takes parameters ...
            """)
            
    with st.expander("View source code"):
        code = inspect.getsource(scl90_gsi_prae_data_eda)
        st.code(code, language='python')
        
    st.subheader(f"Distribution of {selected_variable}")
    scl90_gsi_prae_data, missing_values = scl90_gsi_prae_data_eda(long_format_dataset)
    
    plot_data = [scl90_gsi_prae_data['SCL90_GSI_Prae']]
    group_labels = ["SCL90_GSI_Prae"]
    
    fig = px.box(
    scl90_gsi_prae_data, 
    y='SCL90_GSI_Prae', 
    points="all",  # Shows every individual patient dot next to the box
    boxmode="overlay",
    color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(yaxis_range=[0, 4], yaxis_title="GSI Score (Pre-Treatment)")
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_variable == "SCL90_GSI_Post":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `scl90_gsi_post_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(scl90_gsi_post_data_eda)
        st.code(code, language='python')

    st.subheader(f"Distribution of {selected_variable}")
    scl90_gsi_post_data, missing_values = scl90_gsi_post_data_eda(long_format_dataset)
    
    plot_data = [scl90_gsi_post_data['SCL90_GSI_Post']]
    group_labels = ["SCL90_GSI_Post"]
    
    fig = px.box(
    scl90_gsi_post_data, 
    y='SCL90_GSI_Post', 
    points="all",  # Shows every individual patient dot next to the box
    boxmode="overlay",
    color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(yaxis_range=[0, 4], yaxis_title="GSI Score (Post-Treatment)")
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_variable == "Schulabschluss":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `schulabschluss_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(schulabschluss_data_eda)
        st.code(code, language='python')

    st.subheader(f"Distribution of {selected_variable}")
    schulabschluss_data, schulabschluss_col, missing_values = schulabschluss_data_eda(long_format_dataset)
    fig = px.pie(
        schulabschluss_data, 
        names='Schulabschluss', 
        color='Schulabschluss', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    
    fig.update_layout(showlegend=False)
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "TherapieStatus_PP":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `therapie_status_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(therapie_status_data_eda)
        st.code(code, language='python')
    st.subheader(f"Distribution of {selected_variable}")
    therapie_status_data, therapie_status_col, missing_values = therapie_status_data_eda(long_format_dataset)
    fig = px.pie(
        therapie_status_data, 
        names='TherapieStatus_PP', 
        color='TherapieStatus_PP', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    
    fig.update_layout(showlegend=False)
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "age":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `age_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(age_data_eda)
        st.code(code, language='python')
    st.subheader(f"Distribution of {selected_variable}")
    age_dataset, age_data, outliers, missing_values = age_data_eda(long_format_dataset)
    fig = px.box(age_dataset, age_data, color_discrete_sequence=['#3b766c'])
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif selected_variable == "full_remission":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
                *Function:* `full_remission_data_eda()`
                 Takes parameters ...
            """)
    with st.expander("View source code"):
        code = inspect.getsource(full_remission_data_eda)
        st.code(code, language='python')
    st.subheader(f"Distribution of {selected_variable}")
    full_remission_data, full_remission_col, missing_values = full_remission_data_eda(long_format_dataset)
    full_remission_data['full_remission'] = full_remission_data['full_remission'].map({1: 'Yes', 0: 'No'})
    fig = px.pie(
        full_remission_data, 
        names='full_remission', 
        color='full_remission', 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    
    fig.update_layout(showlegend=False)
    st.metric(
        label="Missing values:",
        value=missing_values    
    )
    st.plotly_chart(fig, use_container_width=False)

elif (selected_variable.startswith(("phq", "gad")) and selected_variable not in sum_variables):
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
        *Function:* `time_series_data_eda()`
        
        Takes a long-format dataset, sorts it by ID and date, and filters it for a specific patient.
        """)

    with st.expander("View source code"):
        code = inspect.getsource(time_series_data_eda)
        st.code(code, language="python")

    st.subheader(f"Time series data for {selected_variable}")

    patient_ids = long_format_dataset["Chiffre"].unique()
    selected_id = st.selectbox(label="Patient ID", options=patient_ids)
    time_series_data = time_series_data_eda(long_format_dataset, patient_id=selected_id)

    if not time_series_data.empty:
        fig = px.line(
            time_series_data,
            x="phqdat",
            y=selected_variable,
            title=f"{selected_variable} response progression for Patient {selected_id}",
            markers=True,  
            color_discrete_sequence=['#3b766c']
        )
        fig.update_yaxes(range=[0, 3], dtick=1)
        st.plotly_chart(fig)
    else:
        st.warning("No data available for this patient.")

elif selected_variable == "phqsum":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
        *Function:* `time_series_data_eda()`
        
        Takes a long-format dataset, sorts it by ID and date, and filters it for a specific patient.
        """)

    with st.expander("View source code"):
        code = inspect.getsource(time_series_data_eda)
        st.code(code, language="python")

    st.subheader(f"Time series data for {selected_variable}")

    patient_ids = long_format_dataset["Chiffre"].unique()
    selected_id = st.selectbox(label="Patient ID", options=patient_ids)
    phqvars = [var for var in long_format_dataset.columns if var.startswith('phq0')]
    gadvars = [var for var in long_format_dataset.columns if var.startswith('gad    0')]

    time_series_data = time_series_data_eda(long_format_dataset, patient_id=selected_id)

    if not time_series_data.empty:
        fig = px.line(
            time_series_data,
            x="phqdat",
            y=selected_variable,
            title=f"{selected_variable} response progression for Patient {selected_id}",
            markers=True,  
            color_discrete_sequence=['#3b766c']
        )
        y_max = 3 * len(phqvars)
        fig.update_yaxes(range=[0, y_max], dtick=1)
        st.plotly_chart(fig)
    else:
        st.warning("No data available for this patient.")

elif selected_variable =="gadsum":
    st.text(metadata[selected_variable]["description"])

    with st.expander("View code documentation"):
        st.markdown("""
        *Function:* `time_series_data_eda()`
        
        Takes a long-format dataset, sorts it by ID and date, and filters it for a specific patient.
        """)

    with st.expander("View source code"):
        code = inspect.getsource(time_series_data_eda)
        st.code(code, language="python")

    st.subheader(f"Time series data for {selected_variable}")

    patient_ids = long_format_dataset["Chiffre"].unique()
    selected_id = st.selectbox(label="Patient ID", options=patient_ids)
    gadvars = [var for var in long_format_dataset.columns if var.startswith('gad0')]

    time_series_data = time_series_data_eda(long_format_dataset, patient_id=selected_id)

    if not time_series_data.empty:
        fig = px.line(
            time_series_data,
            x="phqdat",
            y=selected_variable,
            title=f"{selected_variable} response progression for Patient {selected_id}",
            markers=True,  
            color_discrete_sequence=['#3b766c']
        )
        y_max = 3 * len(gadvars)
        fig.update_yaxes(range=[0, y_max+1], dtick=1)
        st.plotly_chart(fig)
    else:
        st.warning("No data available for this patient.")

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    if st.button("Next: Data Filtering", type='primary'):
        st.switch_page("pages/Data_Filtering.py")