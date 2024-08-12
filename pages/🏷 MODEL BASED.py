import streamlit as st
import pandas as pd
import csv

# Set up the Streamlit page
st.title("Analisis Sentimen")

# Load the lexicons
lexicon_positive = dict()
with open('lexicon/lexicon_positive_ver1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
with open('lexicon/lexicon_negative_ver1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])

# Define the sentiment analysis function
def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word_pos in text:
        if word_pos in lexicon_positive:
            score += lexicon_positive[word_pos]
    for word_neg in text:
        if word_neg in lexicon_negative:
            score += lexicon_negative[word_neg]

    if score > 0:
        polarity = 'positif'
    elif score < 0:
        polarity = 'negatif'
    else:
        polarity = 'netral'

    return score, polarity

# Function to apply color and emoji based on sentiment
def sentiment_indicator(sentiment):
    if sentiment == 'positif':
        return '<span style="color:green;">😊 Positif</span>'
    elif sentiment == 'negatif':
        return '<span style="color:red;">😠 Negatif</span>'
    else:
        return '<span style="color:yellow;">😐 Netral</span>'

# Convert sentiment to labeling_polarity
def sentiment_to_labeling_polarity(sentiment):
    if sentiment == 'positif':
        return 1
    elif sentiment == 'negatif':
        return -1
    else:
        return 0

# File upload with multiple file support
uploaded_files = st.file_uploader("Pilih file CSV", type="csv", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.subheader(f"Dataset Yang Diunggah: {uploaded_file.name}")
        st.dataframe(df, use_container_width=True)
        
        if 'prepos_text' in df.columns:
            # Perform sentiment analysis
            results = df['prepos_text'].apply(lambda x: sentiment_analysis_lexicon_indonesia(x.split()))
            results = list(zip(*results))
            df['polarity_score'] = results[0]
            df['sentiment'] = results[1]

            # Add labeling_polarity column
            df['labeling_polarity'] = df['sentiment'].apply(sentiment_to_labeling_polarity)

            # Display sentiment results
            st.subheader(f"Hasil Analisis Sentimen untuk {uploaded_file.name}")
            st.dataframe(df[['komentar_bersih', 'prepos_text', 'polarity_score', 'sentiment', 'labeling_polarity']], use_container_width=True)

            # Display sentiment count with shadow boxes
            st.subheader(f"Jumlah Sentimen untuk {uploaded_file.name}")
            sentiment_counts = df['sentiment'].value_counts()
            
            sentiment_summary = ''
            for sentiment, count in sentiment_counts.items():
                indicator = sentiment_indicator(sentiment)
                sentiment_summary += f"""
                <div style="
                    border: 2px solid #ddd; 
                    border-radius: 8px; 
                    padding: 10px; 
                    margin-bottom: 10px; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                ">
                    {indicator}: <strong>{count}</strong> komentar
                </div>
                """
            
            st.markdown(sentiment_summary, unsafe_allow_html=True)

            # Add a menu for filtering sentiment counts
            sentiment_options = ['Semua', 'positif', 'negatif', 'netral']
            selected_sentiment = st.selectbox("Pilih Jenis Sentimen untuk Difilter:", sentiment_options)

            if selected_sentiment != 'Semua':
                filtered_df = df[df['sentiment'] == selected_sentiment]
                st.subheader(f"Jumlah Sentimen {selected_sentiment.capitalize()} yang Difilter")
                
                # Display filtered sentiment count with shadow box
                filtered_sentiment_count = filtered_df['sentiment'].value_counts()
                filtered_summary = ''
                for sentiment, count in filtered_sentiment_count.items():
                    indicator = sentiment_indicator(sentiment)
                    filtered_summary += f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 8px; 
                        padding: 10px; 
                        margin-bottom: 10px; 
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    ">
                        {indicator}: <strong>{count}</strong> komentar
                    </div>
                    """
                
                st.markdown(filtered_summary, unsafe_allow_html=True)

            else:
                st.subheader(f"Jumlah Semua Sentimen")
                st.write(df['sentiment'].value_counts())

        else:
            st.error(f"Kolom 'prepos_text' tidak ditemukan dalam file CSV yang diunggah: {uploaded_file.name}")
