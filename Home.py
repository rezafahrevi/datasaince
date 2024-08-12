import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import pandas as pd
import os

# Download NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Set up the Streamlit page
st.set_page_config(page_title="CSV Viewer", page_icon="📊")

st.title("DATA MINING / OLAH DATA")

# File upload with multiple file support
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.subheader(f"Uploaded Dataset: {uploaded_file.name}")
        st.dataframe(df, use_container_width=True)
        
        # Display dataset summary
        st.subheader('Dataset Summary')
        summary = {
            'Jumlah Baris': [df.shape[0]],
            'Jumlah Kolom': [df.shape[1]],
            'Tipe Data': [df.dtypes.to_dict()]
        }
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, use_container_width=True)

        # Calculate and display duplicate comments if the column exists
        if 'komentar' in df.columns:
            st.subheader('Jumlah Komentar yang Duplikasi')
            duplicate_comments = df['komentar'].duplicated().sum()
            st.write(f'Jumlah komentar yang duplikasi: {duplicate_comments}')

            # Remove duplicate rows based on the 'komentar' column
            df_unique = df.drop_duplicates(subset='komentar').copy()  # Use copy to avoid SettingWithCopyWarning

            # Display only the 'komentar' column
            st.subheader('Unique Comments')
            st.dataframe(df_unique[['komentar']], use_container_width=True)

            # Display the number of unique comments
            st.subheader('Jumlah Data Sekarang')
            st.write(f'Jumlah data setelah penghapusan duplikasi: {df_unique.shape[0]} baris')
        else:
            st.write('Kolom "komentar" tidak ditemukan dalam dataset.')

        # Load the slang dictionary
        slang_data = pd.read_csv('colloquial-indonesian-lexicon.csv')
        slang_dict = dict(zip(slang_data['slang'], slang_data['formal']))

        # Define the preprocessing function
        def preprocess_text(text, slang_dict):
            # Case folding: ubah teks menjadi huruf kecil
            text = text.lower()
            # Mengganti kata slang dengan kata baku
            words = text.split()
            normalized_words = [slang_dict.get(word, word) for word in words]
            text = ' '.join(normalized_words)
            # Menghilangkan karakter non-alphabetic dan link
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Menghapus spasi yang berlebihan
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Define tokenization function
        def tokenize_text(text):
            return word_tokenize(text)

        # Define stopword removal function
        def remove_stopwords(tokens, stopwords):
            return [word for word in tokens if word not in stopwords]

        # Define stemming function
        def back_to_root(review_text):
            # Join tokens into a single string
            stop_token = ' '.join(review_text)
            # Create a stemmer instance
            stemmer = StemmerFactory().create_stemmer()
            # Apply stemming
            result_stem = stemmer.stem(stop_token)
            return result_stem

        # Add a button for preprocessing
        if st.button('Preprocess Comments'):
            if 'komentar' in df.columns:
                # Preprocess text
                df_unique.loc[:, 'komentar_bersih'] = df_unique['komentar'].apply(lambda x: preprocess_text(x, slang_dict))
                
                # Tokenization
                df_unique.loc[:, 'token'] = df_unique['komentar_bersih'].apply(lambda x: tokenize_text(x))
                
                # Stopword removal
                indonesian_stop = set(stopwords.words('indonesian'))
                df_unique.loc[:, 'stop_review'] = df_unique['token'].apply(lambda x: remove_stopwords(x, indonesian_stop))
                
                # Stemming
                df_unique.loc[:, 'prepos_text'] = df_unique['stop_review'].apply(lambda x: back_to_root(x))
             
                # Remove rows where 'prepos_text' is empty or contains only whitespace
                df_unique = df_unique[df_unique['prepos_text'].str.strip().astype(bool)]

                # Reset index to ensure the table is sequentially ordered
                df_unique = df_unique.reset_index(drop=True)
                
                # Display the cleaned 'prepos_text' column
                st.subheader('Cleaned Preprocessed Comments')
                st.dataframe(df_unique[['komentar_bersih','prepos_text']], use_container_width=True)

                # Display the number of remaining data entries in 'prepos_text'
                st.subheader('Jumlah Data Setelah Penghapusan Missing Values')
                st.write(f'Jumlah data setelah penghapusan missing values: {df_unique.shape[0]} baris')
            else:
                st.write('Kolom "komentar" tidak ditemukan dalam dataset.')

