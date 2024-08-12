import streamlit as st
import pandas as pd
import csv
import plotly.express as px

# Set up the Streamlit page
st.title("Sentiment Analysis")

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
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        elif word in lexicon_negative:
            score -= lexicon_negative[word]

    if score > 0:
        polarity = 'positif'
    elif score < 0:
        polarity = 'negatif'
    else:
        polarity = 'netral'

    return score, polarity

# File upload with multiple file support
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.subheader(f"Uploaded Dataset: {uploaded_file.name}")
        st.dataframe(df, use_container_width=True)
        
        # View dataset with specific columns
        with st.expander(f"🔎 VIEW DATASET: {uploaded_file.name}"):
            # Define default columns if they exist in the uploaded dataset
            available_columns = df.columns.tolist()
            default_columns = [col for col in ["stem_review", "polarity_score", "sentiment"] if col in available_columns]
            
            # Multiselect to show data with default columns (only if they exist)
            showData = st.multiselect(
                "",
                available_columns,
                default=default_columns
            )
            st.dataframe(df[showData], use_container_width=True)

        # Perform sentiment analysis if 'stem_review' is in the dataset
        if 'stem_review' in df.columns:
            results = df['stem_review'].apply(lambda x: sentiment_analysis_lexicon_indonesia(x.split()))
            results = list(zip(*results))
            df['polarity_score'] = results[0]
            df['sentiment'] = results[1]

            # Display sentiment analysis results
            st.subheader(f"Sentiment Analysis Results for {uploaded_file.name}")
            st.dataframe(df[['stem_review', 'polarity_score', 'sentiment']], use_container_width=True)

            # Display sentiment count
            st.subheader(f"Sentiment Count for {uploaded_file.name}")
            sentiment_counts = df['sentiment'].value_counts()
            st.write(sentiment_counts)

            # Create frequency distribution for sentiments
            frequency = sentiment_counts
            percentage_frequency = frequency / len(df) * 100
            cumulative_frequency = frequency.cumsum()
            relative_frequency = frequency / len(df)
            cumulative_relative_frequency = relative_frequency.cumsum()

            # Create summarized table
            summary_table = pd.DataFrame({
                'Frequency': frequency,
                'Percentage Frequency': percentage_frequency,
                'Cumulative Frequency': cumulative_frequency,
                'Relative Frequency': relative_frequency,
                'Cumulative Relative Frequency': cumulative_relative_frequency
            })

            st.dataframe(summary_table, use_container_width=True)

            # Plotting the histogram using Plotly and Streamlit
            fig = px.histogram(df, x='sentiment', category_orders={'sentiment': ['positif', 'netral', 'negatif']},
                               labels={'sentiment': 'Sentiment', 'count': 'Frequency'}, title='Sentiment Distribution')

            st.success("**SENTIMENT DISTRIBUTION GRAPH**")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"The column 'stem_review' is not found in the uploaded CSV file: {uploaded_file.name}")
