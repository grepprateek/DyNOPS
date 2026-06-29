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

st.header("Data Cleaning")

filtered_dataset = pd.read_csv(f"data/dynops_filtered_dataset_{date.today().strftime('%d-%m-%y')}.csv", low_memory=False)

st.text("In this step, the filtered dataset is subject to following processing steps:")
st.markdown("""
    1. The variable `phqdat` is converted from datetime format to date for better readability.
    2. Instead of removing rows for records having different `phqdat` and `gaddat` entries, we consider `phqdat` as the only date column (renamed to `Date`) and remove `gaddat`.
       This can be improved in further versions of the datasets by correcting the date mismatch issue in the database.
    3. All sessions columns are removed: `Anzahl_BE_Post_PEQ_32`, `phqsessions`, `gadsessions`
    4. The columns `instance_phq` and `instance_gad` are removed as they were only required during data filtering.
    5. The outliers in `age` column are converted to `NaN` (missing values) for imputation in the next step.
    6. The variables `SCL90_GSI_Prae` and `SCL90_GSI_Post` are converted into decimal notation.
    7. All categorical variables are converted into `Int64` data type.
    8. Patients with missing values for `full_remission` column are removed.
""")

if st.button("Clean Data", type='primary'):
    
    # convert phqdat into date format dd-mm-yyyy and remove gaddat for consistency
    filtered_dataset['phqdat'] = pd.to_datetime(filtered_dataset['phqdat']).dt.date
    filtered_dataset['Date'] = filtered_dataset['phqdat']
    filtered_dataset = filtered_dataset.drop(columns=["phqdat", "gaddat"])

    # remove Anzahl_BE_Post_PEQ_32 and instance cols
    filtered_dataset = filtered_dataset.drop(columns = ['Anzahl_BE_Post_PEQ_32', 'instance_phq', 'instance_gad', 'phqsessions', 'gadsessions'])

    # age variable
    filtered_dataset.loc[filtered_dataset['age'] <= 0, 'age'] = np.nan
    filtered_dataset['age'] = filtered_dataset['age'].astype('Int64')

    # therapy status
    filtered_dataset['TherapieStatus_PP'] = filtered_dataset['TherapieStatus_PP'].astype('Int64')

    # scl90_gsi_prae and _post
    filtered_dataset['SCL90_GSI_Prae'] = filtered_dataset['SCL90_GSI_Prae'].str.replace(',', '.', regex=False).astype(float).round(2)
    filtered_dataset['SCL90_GSI_Post'] = filtered_dataset['SCL90_GSI_Post'].str.replace(',', '.', regex=False).astype(float).round(2)

    # bdiiisummenwertprae and post
    filtered_dataset['BDIIISummenwertPrae'] = filtered_dataset['BDIIISummenwertPrae'].astype('Int64')
    filtered_dataset['BDIIISummenwertPost'] = filtered_dataset['BDIIISummenwertPost'].astype('Int64')

    # primdiag_kategorie
    filtered_dataset['PrimDiag_Kategorie'] = filtered_dataset['PrimDiag_Kategorie'].astype('Int64')

    # questionnaire columns
    for col in filtered_dataset.columns:
        if col.startswith(('phq', 'gad')):
            filtered_dataset[col] = filtered_dataset[col].astype('Int64')

    # sessions mismatches
    # sessions_mismatch = filtered_dataset["Chiffre"][filtered_dataset['phqsessions'] != filtered_dataset['gadsessions']].drop_duplicates()
    # filtered_dataset = filtered_dataset[~filtered_dataset['Chiffre'].isin(sessions_mismatch)]
    
    # full remission column
    filtered_dataset['full_remission'] = filtered_dataset['full_remission'].astype('Int64')
    filtered_dataset = filtered_dataset.dropna(subset=['full_remission'])

    filtered_dataset = filtered_dataset.sort_values(by=['Chiffre', 'Date']).reset_index(drop=True)

    cols = list(filtered_dataset.columns)
    cols.remove("Date")  
    target_idx = cols.index("phq001") 
    cols.insert(target_idx, "Date")  
    filtered_dataset = filtered_dataset[cols]

    cols = list(filtered_dataset.columns)
    cols.remove("full_remission")
    cols.append("full_remission")
    filtered_dataset = filtered_dataset[cols]

    filtered_dataset.to_csv(f"data/dynops_cleaned_dataset_{date.today().strftime('%d-%m-%y')}.csv", index=False)
    st.dataframe(filtered_dataset)
    st.write(f"Cleaned dataset has {filtered_dataset.shape[0]} rows and {filtered_dataset.shape[1]} columns.")

if st.button("Next: Data Imputation", type='primary'):
    st.switch_page("pages/Data_Imputation.py")