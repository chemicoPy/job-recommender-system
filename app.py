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
nltk.download('omw-1.4')
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
st.subheader("Navigate to side bar to see project info")
st.subheader("See below for options")


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
    This is a Job recommendation web app that uses filtering techniques and Natural Language Processing (NLP)
    to suggest 10 top jobs to user upon entering a specific job/role (and probably other preferences).
    """)

st.sidebar.header("")  # initialize empty space

st.sidebar.markdown(
    """
    ----------
    ## Text data conversion method is "TF-IDF"
    Term Frequency - Inverse Document Frequency (TF-IDF) converts text data to vectors as model can only process numerical data; it weights the word counts by measure of how often they appear in the dataset
    """)

st.sidebar.header("")  # initialize empty space

st.sidebar.markdown(
            
"""
    ----------
    ## NOTE:
    If the Job/your preferences could not be matched with the available jobs, the overview of job data will be returned with their scores all labeled as "0.0" 
    """)


user_input = st.text_input("Enter any job you want recommendation(s) on")
from_user = pd.DataFrame(data=[user_input], columns = ["Text"])
from_user.index=range(len(from_user.index))

st.text("Vectorizing Method: TF-IDF")

#Replace Nan values
df.fillna("", inplace = True)

#Creating the Jobs corpus
df['Text'] = df['Job title'].map(str) + ' ' + df['Salary'] + ' ' + df['Date'] + ' ' + df['Company Location'] + ' ' + df['Company Name'] + ' ' + df['city']+ ' ' + df['state']
#df.head(10)

data = df[['Unnamed: 0', 'Text', 'Job title','Company Name']]
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
  recommendation = pd.DataFrame(columns = ['Job_ID',  'Job title', 'Company Name', 'Accuracy'], dtype=object)
  count = 0
  for i in top:
      recommendation.at[count, 'Job_ID'] = the_data['Job.ID'][i]
      recommendation.at[count, 'Job title'] = the_data['Job title'][i]
      recommendation.at[count, 'Company Name'] = the_data['Company Name'][i]
      recommendation.at[count, 'Accuracy'] =  scores[count]
      count += 1
  return recommendation

#Using TF-IDF for recommendation

top10_tfidf = sorted(range(len(rec1)), key = lambda i: rec1[i], reverse = True)[:10]  
list_scores_tfidf = [rec1[i][0][0] for i in top10_tfidf]
tfidf_recommendation = get_recommendation(top10_tfidf, data, list_scores_tfidf)     #Recommendation with TF-IDF
tfidf_recommendation["Accuracy"] = tfidf_recommendation["Accuracy"].astype(float)
tfidf_recommendation.Accuracy = tfidf_recommendation.Accuracy.round(2)
   
            
# Another vectorizing method that could be of interest is using Count Vectorizer

count_vect = CountVectorizer()

# Fitting and transforming the vectorizer
count_comb = count_vect.fit_transform((data['Text'])) #fitting and transforming the vector
user_count_countvec = count_vect.transform(from_user['Text'])
cos_sim_count_countvec = map(lambda x: cosine_similarity(user_count_countvec, x), count_comb)
count_vec1 = list(cos_sim_count_countvec)   
top10_count_vec_count = sorted(range(len(count_vec1)), key = lambda i: count_vec1[i], reverse = True)[:10]
list_scores_vec_count = [count_vec1[i][0][0] for i in top10_count_vec_count]
count_vec_recommendation = get_recommendation(top10_count_vec_count, data, list_scores_vec_count)   #Recommendation with count vectorizer
     
    
def main():
    
    if st.button("Recommend Jobs"):
        st.write(tfidf_recommendation)

            
if __name__ == '__main__':
    main()

