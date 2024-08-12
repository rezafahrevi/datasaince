import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build

def komentar_vidio(api_key, video_id):
    balasan = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    hasil = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

    while hasil:
        for item in hasil['items']:
            published = item['snippet']['topLevelComment']['snippet']['publishedAt']
            user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']

            balasan.append([published, user, comment, likeCount])
            replycount = item['snippet']['totalReplyCount']

            if replycount > 0:
                for reply in item['replies']['comments']:
                    published = reply['snippet']['publishedAt']
                    user = reply['snippet']['authorDisplayName']
                    repl = reply['snippet']['textDisplay']
                    likeCount = reply['snippet']['likeCount']

                    balasan.append([published, user, repl, likeCount])

        if 'nextPageToken' in hasil:
            hasil = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=hasil['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break

    return balasan

# Streamlit page
st.title("Ambil Komentar dari Video YouTube")

# Input fields
api_key = st.text_input("Masukkan API Key YouTube", "")
video_url = st.text_input("Masukkan URL Video YouTube", "")

if video_url:
    # Extract video ID from URL
    video_id = video_url.split("v=")[-1]
    if "&" in video_id:
        video_id = video_id.split("&")[0]

    if st.button("Ambil Komentar"):
        if api_key and video_id:
            try:
                comments = komentar_vidio(api_key, video_id)
                df = pd.DataFrame(comments, columns=['tanggal_publis', 'nama', 'komentar', 'likeCount'])
                sorted_df = df.sort_values(by='tanggal_publis', ascending=False) # Sort by newest
                sorted_df = sorted_df.reset_index(drop=True)
                st.subheader("Komentar dari Video")
                st.dataframe(sorted_df)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan masukkan API Key dan URL Video.")

# Additional information or instructions
st.markdown("""
### Petunjuk
1. Masukkan API Key YouTube Anda di field "Masukkan API Key YouTube".
2. Masukkan URL video YouTube di field "Masukkan URL Video YouTube".
3. Klik tombol "Ambil Komentar" untuk mengolah komentar dari video yang dipilih.
""")
