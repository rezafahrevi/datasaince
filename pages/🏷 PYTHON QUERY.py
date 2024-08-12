import streamlit as st
import pandas as pd 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score



 
#navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")  

st.header("PREDICTIVE ANALYTICS DASHBOARD")
st.image("data/logo2.webp",caption="")
st.write("MULTIPLE REGRESSION WITH  SSE, SE, SSR, SST, R2, ADJ[R2], RESIDUAL")
st.success("The main objective is to measure if Number of family dependents and Wives may influence a person to supervise many projects")
 
# load CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#logo

st.sidebar.title("PREDICT NEW VALUES")

df = pd.read_csv('dataset_bersihindoax1.csv')

X_train, X_test, y_train, y_test = train_test_split(df['stemming_komentar'], df['label'],
                                                    test_size = 0.20,
                                                    random_state = 0)

tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
y_pred = nb.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)

clf = MultinomialNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

st.write("MultinomialNB Accuracy:", accuracy_score(y_test,predicted))
st.write("MultinomialNB Precision:", precision_score(y_test,predicted, average="weighted", pos_label="positif"))
st.write("MultinomialNB Recall:", recall_score(y_test,predicted, average="weighted", pos_label="Negatif"))
st.write("MultinomialNB f1_score:", f1_score(y_test,predicted, average="weighted", pos_label="Netral"))
st.write(f'confusion_matrix:\n {confusion_matrix(y_test, predicted)}')
st.write('====================================================\n')
st.write(classification_report(y_test, predicted, zero_division=0))

def confusion_matrix(metrics_list):
   if 'Confusion Matrix' in metrics_list:
            st.subheader({confusion_matrix(y_test, predicted)})         
            st.pyplot()
            
# Fit a linear regression model
  # Display prediction
st.sidebar.image("data/logo1.png",caption="")
















 
