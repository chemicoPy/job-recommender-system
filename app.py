
import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv("./clean_data.csv")

# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Job Recommender System')
st.subheader("See below for options")

# re-configuring page layout to restrict users from overwriting the app configuraion

hide_streamlit_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
'''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.sidebar.markdown(
            """
     ----------
    ## Project Overview
    This is an Job recommendation web app that can recommend 10 top jobs to user upon entering a specific job/role.    
    """)

st.sidebar.header("")  # initialize empty space

st.sidebar.markdown(
    """
    ----------
    ## You can either select:
    1. TF-IDF
    2. Count Vectorizer
     
    """)

text = st.text_input("Enter any job you want recommendation(s) on")
model_select = st.selectbox(
            "Select a model you'd like to work with",
            (
                "TF-IDF",
                "Count Vectorizer"
            ),
        )


if st.button("convert"):
    
    if model_select == "TF-IDF":
        st.write(df)
        
    elif model_select == "Count Vectorizer":
        st.write(df)

    


# In[ ]:




