import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

def filter_and_save_csv(input_df, column_name, keywords):
    filtered_dfs = {}
    for keyword in keywords:
        if isinstance(keyword, tuple):
            # Jika keyword adalah tuple, gabungkan kata-kata dalam tuple menjadi satu pola pencarian
            pattern = '|'.join(keyword)
        else:
            pattern = keyword

        filtered_df = input_df[input_df[column_name].str.contains(pattern, case=False, na=False)]
        filtered_df = filtered_df.reset_index(drop=True)  # Reset index
        
        # Hapus kolom yang tidak diinginkan
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.contains('^Unnamed')]
        
        filtered_dfs[keyword] = filtered_df
    return filtered_dfs

def hitung_persen_positif(jumlah_positif, total_kata_positif):
    if total_kata_positif > 0:
        return (jumlah_positif / total_kata_positif) * 100
    return 0

# Streamlit UI
st.title('Filter dan Tampilkan Data CSV')

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Input kolom dan kata kunci
    column_name = st.text_input('Nama Kolom untuk Filter', 'prepos_text')

    # Input manual kata kunci
    keyword_input = st.text_area(
        'Masukkan daftar kata kunci, pisahkan dengan koma atau gunakan format tuple',
        value="('anies', 'anis'), 'Prabowo', 'ganjar'"
    )
    try:
        # Evaluasi input dari string menjadi list
        keywords = eval(keyword_input)
    except:
        st.error('Format input kata kunci tidak valid. Gunakan format tuple atau string yang dipisahkan koma.')
        keywords = []

    # Filter data dan simpan output
    if st.button('Filter Data'):
        if uploaded_file:
            filtered_dfs = filter_and_save_csv(df, column_name, keywords)
            st.session_state.filtered_dfs = filtered_dfs
            st.session_state.column_name = column_name

            for keyword, filtered_df in filtered_dfs.items():
                st.write(f"### Output untuk keyword: '{keyword}'")
                st.dataframe(filtered_df)  # Menampilkan seluruh data hasil yang difilter

                # Menampilkan jumlah sentiment
                if 'sentimen_NaiveBayes' in filtered_df.columns:
                    sentiment_counts = filtered_df['sentimen_NaiveBayes'].value_counts()
                    st.write(f"Sentiment counts:")
                    st.write(f"😊 **Positif**: {sentiment_counts.get('positif', 0)} komentar")
                    st.write(f"😠 **Negatif**: {sentiment_counts.get('negatif', 0)} komentar")
                    st.write(f"😐 **Netral**: {sentiment_counts.get('netral', 0)} komentar")
                else:
                    st.write(f"Kolom 'sentimen_NaiveBayes' tidak ditemukan di output")

    # Hitung persentase jika tombol ditekan
    if st.button('Hitung Persentase'):
        if 'filtered_dfs' in st.session_state:
            filtered_dfs = st.session_state.filtered_dfs
            column_name = st.session_state.column_name

            persen_positif_list = []
            labels = []
            total_kata_positif = 0
            jumlah_positif_list = []

            for keyword, filtered_df in filtered_dfs.items():
                # Hitung jumlah dan total kata positif
                if 'sentimen_NaiveBayes' in filtered_df.columns:
                    sentiment_counts = filtered_df['sentimen_NaiveBayes'].value_counts()
                    jumlah_positif = sentiment_counts.get('positif', 0)
                    total_kata_positif += sentiment_counts.sum()
                    jumlah_positif_list.append(jumlah_positif)
                    labels.append(keyword)
                else:
                    st.write(f"Kolom 'sentimen_NaiveBayes' tidak ditemukan di output")

            # Hitung persentase positif untuk setiap keyword
            persen_positif_list = [hitung_persen_positif(jumlah, total_kata_positif) for jumlah in jumlah_positif_list]

            # Plot diagram lingkaran jika ada data yang valid
            if persen_positif_list:
                sizes = persen_positif_list
                colors = ['blue', 'lightskyblue', 'red'][:len(sizes)]  # Warna sesuai jumlah kategori
                explode = (0.1,) + (0,) * (len(sizes) - 1)  # untuk memberikan efek 'explode' pada slice pertama
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 14})
                ax.axis('equal')  # Memastikan lingkaran terlihat sebagai lingkaran
                plt.title('Persentase Sentiment Positif (Naive Bayes)', fontsize=16)
                
                # Simpan gambar ke buffer dan tampilkan di Streamlit
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close(fig)
            else:
                st.write("Tidak ada data positif untuk ditampilkan.")
        else:
            st.write("Silakan filter data terlebih dahulu.")
