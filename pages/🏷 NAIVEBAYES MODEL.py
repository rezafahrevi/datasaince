import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set up the Streamlit page
st.title("Prediksi Sentimen Naive Bayes")

# File upload
uploaded_file = st.file_uploader("Pilih file CSV untuk pelatihan", type="csv")

if uploaded_file:
    # Membaca data dari CSV yang diunggah
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset yang Diupload")
    st.dataframe(data, use_container_width=True)  # Menampilkan semua baris jika memungkinkan

    # Memisahkan fitur dan label
    X = data['prepos_text']  # Kolom yang berisi teks
    y = data['labeling_polarity']  # Kolom yang berisi label sentimen (1 = positif, -1 = negatif, 0 = netral)

    # Memilih persentase data uji
    test_size_percentage = st.slider("Pilih Persentase Data Uji", min_value=10, max_value=100, step=10)
    test_size = test_size_percentage / 100

    # Memisahkan data untuk pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Mengubah teks menjadi fitur TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Membuat dan melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Memprediksi pada data uji
    y_pred = model.predict(X_test_tfidf)

    # Menghitung confusion matrix dan metrik evaluasi
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Menampilkan hasil evaluasi
    st.subheader("Hasil Evaluasi")

    # Display metrics in a styled format
    total1, total2, total3, total4 = st.columns(4, gap='small')
    
    with total1:
        st.info('Akurasi', icon="✅")
        st.metric(label="Akurasi", value=f"{accuracy:.2f}")

    with total2:
        st.info('Presisi', icon="📏")
        st.metric(label="Presisi", value=f"{precision:.2f}")

    with total3:
        st.info('Recall', icon="🔍")
        st.metric(label="Recall", value=f"{recall:.2f}")

    with total4:
        st.info('F1 Score', icon="⚖️")
        st.metric(label="F1 Score", value=f"{f1:.2f}")

    # Visualisasi Confusion Matrix menggunakan Matplotlib
    st.subheader("Visualisasi Matriks Kebingungan")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

    # Fungsi untuk memprediksi sentimen dari file CSV baru
    def predict_sentiment_from_csv(input_csv, output_csv):
        new_data = pd.read_csv(input_csv)
        new_data_tfidf = tfidf.transform(new_data['prepos_text'])
        new_data_predictions = model.predict(new_data_tfidf)
        
        # Mendefinisikan label sentimen
        sentiment_labels = {1: 'Positif', -1: 'Negatif', 0: 'Netral'}
        
        # Mengubah prediksi numerik menjadi label sentimen
        new_data['predicted_sentiment'] = [sentiment_labels.get(pred, 'Tidak Diketahui') for pred in new_data_predictions]
        
        # Menyimpan hasil prediksi ke file CSV
        new_data.to_csv(output_csv, index=False)
        return new_data

    # Upload file baru untuk prediksi
    st.subheader("Prediksi Sentimen pada Data Baru")
    new_uploaded_file = st.file_uploader("Pilih file CSV untuk prediksi", type="csv")
    output_filename = st.text_input("Nama file output", "prediksi_sentimen.csv")

    if new_uploaded_file and st.button("Prediksi Sentimen"):
        predicted_data = predict_sentiment_from_csv(new_uploaded_file, output_filename)
        st.success(f"Prediksi sentimen selesai dan disimpan ke {output_filename}")
        st.dataframe(predicted_data, use_container_width=True)  # Menampilkan semua baris jika memungkinkan

        # Jika file yang diunggah untuk prediksi memiliki kolom 'labeling_polarity', hitung Confusion Matrix dan metrik lainnya
        if 'labeling_polarity' in predicted_data.columns:
            st.subheader("Metrik Evaluasi untuk Prediksi Baru")
            y_true = predicted_data['labeling_polarity']
            y_pred_new = predicted_data['predicted_sentiment'].map({'Positif': 1, 'Negatif': -1, 'Netral': 0})

            cm_new = confusion_matrix(y_true, y_pred_new)
            accuracy_new = accuracy_score(y_true, y_pred_new)
            precision_new = precision_score(y_true, y_pred_new, average='weighted')
            recall_new = recall_score(y_true, y_pred_new, average='weighted')
            f1_new = f1_score(y_true, y_pred_new, average='weighted')

            st.write(f"Akurasi: {accuracy_new:.2f}")
            st.write(f"Presisi: {precision_new:.2f}")
            st.write(f"Recall: {recall_new:.2f}")
            st.write(f"F1 Score: {f1_new:.2f}")

            # Visualisasi Confusion Matrix untuk data baru
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_new, annot=True, fmt="d", cmap="Blues", xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(plt)

    # Input manual untuk prediksi sentimen
    st.subheader("Prediksi Sentimen untuk Teks Baru")

    # Menambahkan CSS untuk mengubah warna kotak input
    st.markdown("""
        <style>
        .css-1pahdxg {
            background-color: #f0f0f0;  /* Warna abu-abu */
            color: #000000;  /* Warna teks hitam untuk kontras */
        }
        </style>
        """, unsafe_allow_html=True)

    new_text = st.text_area("Masukkan teks untuk prediksi", "")

    if st.button("Prediksi Sentimen dari Teks"):
        if new_text:
            # Fungsi untuk memprediksi sentimen dari teks baru
            def predict_sentiment(new_text):
                new_text_tfidf = tfidf.transform([new_text])
                prediction = model.predict(new_text_tfidf)[0]
                sentiment_labels = {1: 'Positif', -1: 'Negatif', 0: 'Netral'}
                return sentiment_labels.get(prediction, 'Tidak Diketahui')

            prediction = predict_sentiment(new_text)
            
            # Menambahkan emoji sesuai dengan prediksi
            if prediction == 'Positif':
                st.write(f"Prediksi Sentimen: {prediction} 😀")
            elif prediction == 'Negatif':
                st.write(f"Prediksi Sentimen: {prediction} 😡")
            elif prediction == 'Netral':
                st.write(f"Prediksi Sentimen: {prediction} 😐")
            else:
                st.write(f"Prediksi Sentimen: {prediction}")
        else:
            st.warning("Silakan masukkan teks sebelum menekan tombol prediksi.")
