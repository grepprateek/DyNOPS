import streamlit as st
st.set_page_config(page_title="About", layout="centered")
st.header("DyNOPS: Dynamic Networks of Outpatient Psychotherapy Symptoms")

col1, col2 = st.columns([1, 2])
with col1:
    st.image("DyNOPS.png", link="https://www.tu-braunschweig.de/psychologie/psychotherapieambulanz/team/dynops-der-dino")
with col2:
    st.text("DyNOPS (Dynamic Bayesian Network Modelling of Psychopathology during Outpatient Psychotherapy) is a collaborative research project between the Institute of Psychology and the Institute of Data Science in Biomedicine at TU Braunschweig. The project aims to view psychological disorders as causal interactions between symptoms forming a continuous feedback loop, rather than as observable parts of a latent condition. DyNOPS uses dynamic Bayesian networks to understand how symptoms of depression and anxiety disorders are interconnected during outpatient psychotherapy.", text_alignment="justify")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
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
    if st.button("Get started", type='primary'):
        st.switch_page("pages/Data_Reshaping.py")
