import streamlit as st
import pandas as pd
import os

def user_page():
    st.title('Halaman Pengguna')
    
    # Direktori tempat menyimpan data yang telah diunduh
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Menampilkan file yang telah diunduh sebelumnya
    st.subheader('Data yang telah diunduh sebelumnya:')
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if files:
        for file in files:
            st.write(file)
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                st.write(df.head())
            except Exception as e:
                st.write(f'Gagal membaca file {file}: {e}')
    else:
        st.write('Belum ada data yang diunduh sebelumnya.')

    # Mengunggah data baru
    st.subheader('Unggah Data Baru:')
    uploaded_file = st.file_uploader('Pilih file CSV', type='csv')
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write('Data yang diunggah:')
            st.write(new_data.head())
            
            # Menyimpan file yang diunggah
            save_path = os.path.join(data_dir, uploaded_file.name)
            new_data.to_csv(save_path, index=False)
            st.success(f'File berhasil diunggah dan disimpan sebagai {uploaded_file.name}')
        except Exception as e:
            st.error(f'Gagal mengunggah file: {e}')

if __name__ == "__main__":
    user_page()
