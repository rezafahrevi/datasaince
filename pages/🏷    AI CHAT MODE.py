import streamlit as st
import re

# Function to extract sentences based on conjunctions and entities
def extract_sentences_with_conjunctions(text, entities):
    # Split the text into sentences based on conjunctions and punctuation
    sentences = re.split(r'\s*(?:,|;|(?<!\w)\b(?:tapi|sementara|namun|sebaliknya|begitu|walaupun|meskipun)\b)\s*', text)
    # Filter sentences that contain any of the named entities
    filtered_sentences = [sentence.strip() for sentence in sentences if any(entity in sentence.lower() for entity in entities)]
    return filtered_sentences

# Named entities to look for
entities = ["prabowo", "ganjar", "anies", "anis", "gibran", "imin", "mahfud"]

# Streamlit application
st.title("Ekstraksi Kalimat Berdasarkan Entitas")

# Text input
text = st.text_area("Masukkan teks:", height=200)

if st.button("Ekstrak Kalimat"):
    if text:
        # Extract sentences based on conjunctions
        sentences = extract_sentences_with_conjunctions(text, entities)
        
        # Display the extracted sentences
        st.subheader("Kalimat yang Mengandung Entitas:")
        if sentences:
            for i, sentence in enumerate(sentences, 1):
                st.write(f"Kalimat {i}: {sentence}")
        else:
            st.write("Tidak ada kalimat yang mengandung entitas yang ditentukan.")
    else:
        st.write("Silakan masukkan teks untuk analisis.")
