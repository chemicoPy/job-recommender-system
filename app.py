import pandas as pd
import numpy as np
import nltk
#nltk.download()
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import streamlit as st




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

user_input = st.text_input("Enter any job you want recommendation(s) on")
from_user = pd.DataFrame(data=[user_input], columns = ["Text"])
from_user.index=range(len(from_user.index))

model_select = st.selectbox(
            "Select a model you'd like to work with",
            (
                "TF-IDF",
                "Count Vectorizer"
            ),
        )

#Replace Nan values
df.fillna("", inplace = True)

#Creating the Jobs corpus
df['Text'] = df['Job title'].map(str) + ' ' + df['Salary'] + ' ' + df['Date'] + ' ' + df['Company Location'] + ' ' + df['Company Name'] + ' ' + df['city']+ ' ' + df['state']
#df.head(10)

data = df[['Unnamed: 0', 'Text', 'Job title']]
data = data.fillna(' ')
data.rename(columns={'Unnamed: 0':"Job.ID"}, inplace = True)

stopword = stopwords.words('english')
stopword_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Create word tokens
def token_txt(token):
    return token not in stopword_ and token not in list(string.punctuation) and len(token) > 2   
  
def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "", text)
  text = re.sub("(\\d|\\W)+", " ", text) 
  text = text.replace("nbsp", "")
  clean_text = [wn.lemmatize(word, pos = "v") for word in word_tokenize(text.lower()) if token_txt(word)]
  clean_text2 = [word for word in clean_text if token_txt(word)]

  return " ".join(clean_text2)


data['Text'] = data['Text'].apply(clean_txt) 

tfidf_vect = TfidfVectorizer()

# Fitting and transforming the vector
tfidf_comb = tfidf_vect.fit_transform((data['Text'])) #Computing the Cosine Similarity using TF-IDF

user_tfidf = tfidf_vect.transform(from_user['Text'])
cos_sim_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_comb)
rec1 = list(cos_sim_tfidf)

def get_recommendation(top, the_data, scores):
  recommendation = pd.DataFrame(columns = ['Job_ID',  'Job title', 'Score'], dtype=object)
  count = 0
  for i in top:
      recommendation.at[count, 'Job_ID'] = the_data['Job.ID'][i]
      recommendation.at[count, 'Job title'] = the_data['Job title'][i]
      recommendation.at[count, 'Score'] =  scores[count]
      count += 1
  return recommendation

#Using TF-IDF for recommendation

top10_tfidf = sorted(range(len(rec1)), key = lambda i: rec1[i], reverse = True)[:10]  
list_scores_tfidf = [rec1[i][0][0] for i in top10_tfidf]
tfidf_recommendation = get_recommendation(top10_tfidf, data, list_scores_tfidf)     #Recommendation with TD-IDF
    
   
#Using Count Vectorizer for recommendation

count_vect = CountVectorizer()

# Fitting and transforming the vectorizer
count_comb = count_vect.fit_transform((data['Text'])) #fitting and transforming the vector
user_count_countvec = count_vect.transform(from_user['Text'])
cos_sim_count_countvec = map(lambda x: cosine_similarity(user_count_countvec, x), count_comb)
count_vec1 = list(cos_sim_count_countvec)   
top10_count_vec_count = sorted(range(len(count_vec1)), key = lambda i: count_vec1[i], reverse = True)[:10]
list_scores_vec_count = [count_vec1[i][0][0] for i in top10_count_vec_count]
count_vec_recommendation = get_recommendation(top10_count_vec_count, data, list_scores_vec_count)   #Recommendation with count_vec_recommendation
     
    
def main():
    
    if st.button("convert"):
        if model_select == "TF-IDF":
            st.write(tfidf_recommendation)
            
        elif model_select == "Count Vectorizer":
            st.write(count_vec_recommendation)



if __name__ == '__main__':
    main()



