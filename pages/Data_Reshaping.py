import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path
import inspect
from pandas._typing import FilePath

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

st.header("Data Reshaping")
st.markdown("""
Converts the provided **wide-format** dataset into **long-format** by unpivoting the dataframe, 
making it compatible for predictive modeling algorithms.
""")

def convert_wide_to_long(
        source_dataset_path: FilePath = None,
        separator: str = ';',
        id_column: str = None,
        phq_columns_prefixes: list = None,
        gad_columns_prefixes: list = None,
        wide_format_dataset_path: FilePath = None,
        long_format_dataset_path: FilePath = None
) -> pd.DataFrame:
    """
    Converts the provided wide-format dataset into long-format dataset by unpivoting the dataframe, making it compatible for predictive modeling algorithms.

    Attributes
    ----------
    source_dataset_path: FilePath
        Path to the initial dataset in wide-format sourced from the data management software psychoEQ.
    separator: str
        Symbol separating values in the wide-format dataset, such as `';'`(default) or `','`.
    id_column: str
        ID column in the dataset serving as the primary key assigned to each unique patient.
    phq_column_prefixes: list
        Depending on the number of questionnaire sets and other associated variables, input a list that contains time-dependent columns for PHQ. 
        For e.g., `['phqdat', 'instance_phq', 'phq001', 'phq002', 'phq003', 'phq004', 'phq005', 'phq006', 'phq007', 'phq008', 'phq009', 'phq010', 'phq011', 'phqo00', 'phqsum']`.
    gad_column_prefixes: list
        Depending on the number of questionnaire sets and other associated variables, input a list that contains time-dependent columns for GAD. 
        For e.g., `['gaddat', 'instance_gad', 'gad001', 'gad002', 'gad003', 'gad004', 'gad005', 'gad006', 'gad007', 'gad008', 'gad009', 'gad010', 'gad011', 'gado00', 'gadsum']`.
    wide_format_dataset_path: FilePath
        Path to save the wide-format dataset as a readable CSV file.
    long_format_dataset_path: FilePath
        Path to save the long-format version of the original dataset as a CSV file.

    Returns
    -------
    long_format_dataset: pd.DataFrame
    """
    source_dataset = pd.read_csv(source_dataset_path, sep=separator, engine='python')
    source_dataset.to_csv(Path(wide_format_dataset_path))
    print(f"A readable version of the provided wide format dataset is saved at: {wide_format_dataset_path}")

    columns_prefixes = phq_columns_prefixes + gad_columns_prefixes

    # Unpivots the dataframe
    long_format_dataset = pd.wide_to_long(
        df = source_dataset,
        stubnames = columns_prefixes,
        i = id_column,
        j = 'session_num',
        sep = '_'
    ).reset_index()

    long_format_dataset = long_format_dataset.sort_values(by = [id_column, 'session_num']).reset_index(drop=True)
    long_format_dataset.to_csv(Path(long_format_dataset_path), index=False)
    print(f"The wide format dataset is converted to long format and saved at: {long_format_dataset_path}")

    return long_format_dataset


with st.expander("Read documentation"):
    st.markdown("""
        At Psychotherapieambulanz (PTA), the dataset is managed by two data management tools: psychoPlan (for patient management) and psychoEQ (for questionnaires and data management).
        The unprocessed data is stored in a wide-format dataset, with each column representing the time-dependent and -independent variables, with the former being suffixed with therapy session number.
        Each row represents the entire time-series data for a single patient, spread across the therapy sessions.\n
        In order to make the dataset compatible for learning temporal dependencies, it is converted into a long format with the following changes:
        1. All timestamps are stored into a single column and the rows are pivoted such that each time-dependent column value turns into a row entry for the corresponding timestamp.
        2. In doing so, each column represents a unique variable and has values sequenced by timestamp.
        3. For static variables, including 'Chiffre' (ID column), the row entries are duplicated.
        4. A separate `session_num` column is created indicating the therapy session number.\n    
        The resultant long-format dataset is stored as a CSV file in the `\data` directory.
    """)

with st.expander("View Source Code"):
    code = inspect.getsource(convert_wide_to_long)
    st.code(code, language='python')

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Wide-Format CSV", type=["csv"])
    separator = st.selectbox("Separator", [";", ","], index=0)
    id_col = st.text_input("ID Column", value="Chiffre")

with col2:
    st.subheader("2. Column Prefixes")
    phq_input = st.text_area(
    "PHQ Prefixes (comma separated)",
    value="phqdat, instance_phq, phq001, phq002, phq003, phq004, phq005, phq006, phq007, phq008, phq009, phqsum"
    )

    gad_input = st.text_area(
        "GAD Prefixes (comma separated)",
        value="gaddat, instance_gad, gad001, gad002, gad003, gad004, gad005, gad006, gad007, gad008, gad009, gadsum"
    )

if uploaded_file is not None:
    try:
        dynops_dataset = pd.read_csv(uploaded_file, sep=separator, low_memory=False)
        st.write("### Data Preview (Wide)")
        st.write(f"Number of patients (rows) in the dataset: {dynops_dataset.shape[0]}")
        st.write(f"Number of columns in the dataset: {dynops_dataset.shape[1]}")
        st.dataframe(dynops_dataset)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    phq_list = [x.strip() for x in phq_input.split(",") if x.strip()]
    gad_list = [x.strip() for x in gad_input.split(",") if x.strip()]

    st.write("Click on the following button to reshape the uploaded data into long-format.")
    if st.button("Reshape Data", type="primary"):
        try:
            if id_col not in dynops_dataset.columns:
                st.error(f"ID column '{id_col}' not found in dataset.")
                st.stop()

            all_prefixes = phq_list + gad_list
            missing_prefixes = []
            for prefix in all_prefixes:
                matching_cols = [col for col in dynops_dataset.columns if col.startswith(prefix + "_")]

                if not matching_cols:
                    missing_prefixes.append(prefix)

            if missing_prefixes:
                st.error(f"No matching columns found for prefixes: {missing_prefixes}")
                st.stop()

            with st.spinner("Reshaping data... until then feel free to take a sip or two of your coffee"):
                uploaded_file.seek(0)
                wide_format_dataset = convert_wide_to_long(
                    source_dataset_path=uploaded_file,
                    separator=separator,
                    id_column=id_col,
                    phq_columns_prefixes=phq_list,
                    gad_columns_prefixes=gad_list,
                    wide_format_dataset_path = f"data/dynops_wide_format_dataset_{date.today().strftime('%d-%m-%y')}.csv",
                    long_format_dataset_path = f"data/dynops_long_format_dataset_{date.today().strftime('%d-%m-%y')}.csv"
                )

            st.success("Transformation Successful!")
            st.write("### Data Preview (Long)")
            st.write(f"The long format dataset has {wide_format_dataset.shape[0]} rows and {wide_format_dataset.shape[1]} columns.")
            st.dataframe(wide_format_dataset)

            wide_format_dataset.to_csv(index=False).encode("utf-8")

        except Exception as e:
            st.error(f"Transformation Error: {e}")

            st.info(
                "Check if:\n"
                "- ID column exists\n"
                "- Prefixes match column names\n"
                "- CSV separator is correct"
            )
    
    if st.button("Next: Exploratory Data Analysis", type="primary"):
        st.switch_page("pages/Exploratory_Data_Analysis.py")

# Modify the following code for standalone script implementation.

# if __name__ == "__main__":
#     dynops_dataset_path = "data/raw/export-to-idsb_2026-04-02.csv"
#     id_column = 'Chiffre'
#     phq_columns_prefixes = ['phqdat', 'instance_phq', 'phq001', 'phq002', 'phq003', 'phq004', 'phq005', 'phq006', 'phq007', 'phq008', 'phq009', 'phq010', 'phq011', 'phqo00', 'phqsum']
#     gad_columns_prefixes = ['gaddat', 'instance_gad', 'gad001', 'gad002', 'gad003', 'gad004', 'gad005', 'gad006', 'gad007', 'gad008', 'gad009', 'gad010', 'gad011', 'gado00', 'gadsum']
#     wide_format_dataset_path = f"data/processing/dynops_wide_format_dataset_{date.today().strftime("%d-%m-%y")}.csv"
#     long_format_dataset_path = f"data/processing/dynops_long_format_dataset_{date.today().strftime("%d-%m-%y")}.csv"
#     convert_wide_to_long(dynops_dataset_path, ';', 'Chiffre', phq_columns_prefixes, gad_columns_prefixes, wide_format_dataset_path, long_format_dataset_path)
