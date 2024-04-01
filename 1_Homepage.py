import streamlit as st
import cv2

st.set_page_config(
    page_title="Traffic Monitoring App",
    page_icon="",layout="wide"
)

st.markdown("<h1 style='text-align: center; color: White;'>Smart Traffic Management System</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: lightblue;'>National Centre of Big Data & Cloud Computing</h4>", unsafe_allow_html=True)
sample_img =cv2.imread('ncbc2.jpg')
FRAME_WINDOW=st.image(sample_img,channels='BGR')