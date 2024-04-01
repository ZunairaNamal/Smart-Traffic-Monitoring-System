import streamlit as st
import cv2
import torch
from utils.hubconf import custom1,custom2
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
import math

p_time = 0
oldfps=1
fps=1

sample_img = cv2.imread('ncbcc.png')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('Settings')


path_model_file='best (3).pt'

path_to_class_txt=['Bus','Car','Motor Cycle','Person','Rickshaw','Truck']
if path_to_class_txt is not None:

    options = st.sidebar.radio(
        'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)

    gpu_option = st.sidebar.radio(
        'PU Options:', ('GPU', 'CPU'))

    if not torch.cuda.is_available():
        st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
    else:
        st.sidebar.success(
            'GPU is Available on this Device, Choose GPU for the best performance',
            icon="✅"
        )

    # Confidence
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=3
    )
 
    class_labels = path_to_class_txt
    color_pick_list =  [[255, 0, 255], [0, 0, 255], [0, 0, 0], [0, 255, 0], [255, 0, 0], [205, 127, 127]]
   
    # Image
    if options == 'Image':
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            pred = st.checkbox('Predict Using YOLOv7')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')

            if pred:
                if gpu_option == 'CPU':
                    model = custom1(path_or_model=path_model_file)
                if gpu_option == 'GPU':
                    model = custom2(path_or_model=path_model_file, gpu=True)
                
                bbox_list = []
                current_no_class = []
                results = model(img)
                
                # Bounding Box
                box = results.pandas().xyxy[0]
                class_list = box['class'].to_list()

                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])
                if len(bbox_list) != 0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, img, label=class_labels[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])
                FRAME_WINDOW.image(img)

    # Video
    if options == 'Video':
        upload_video_file = st.sidebar.file_uploader(
            'Upload Video', type=['mp4', 'avi', 'mkv'])
        if upload_video_file is not None:
            pred = st.checkbox('Predict Using YOLOv7')
            # Model
            if gpu_option == 'CPU':
                model = custom1(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom2(path_or_model=path_model_file, gpu=True)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(upload_video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            if pred:
                FRAME_WINDOW.image([])
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            'Video file NOT working\n \
                            Check Video path or file properly!!',
                            icon="🚨"
                        )
                        break
                    current_no_class = []
                    bbox_list = []
                    results = model(img)
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    FRAME_WINDOW.image(img)
                    
                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    
                    # Current number of classes\

    # Web-cam
    if options == 'Webcam':
        cam_options = st.sidebar.selectbox('Webcam Channel',
                                           ('Select Channel', '0', '1', '2', '3'))
        # Model
        if gpu_option == 'CPU':
            model = custom1(path_or_model=path_model_file)
        if gpu_option == 'GPU':
            model = custom2(path_or_model=path_model_file, gpu=True)

        if len(cam_options) != 0:
            if not cam_options == 'Select Channel':
                cap = cv2.VideoCapture(int(cam_options))
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f'Webcam channel {cam_options} NOT working\n \
                            Change channel or Connect webcam properly!!',
                            icon="🚨"
                        )
                        break

                    bbox_list = []
                    current_no_class = []
                    results = model(img)
                    
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id],
                                         color=color_pick_list[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    FRAME_WINDOW.image(img, channels='BGR')

                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    with stframe3.container():
                        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                        js1, js2, js3 = st.columns(3)   
                                      

                        # Updating System stats
                        with js1:
                            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                            mem_use = psutil.virtual_memory()[2]
                            if mem_use > 50:
                                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)



    # RTSP
    if options == 'RTSP':

        rtsp_url =st.sidebar.selectbox('Camera IP',
                                           ('rtsp://admin:ncbc1234@192.168.1.156', 
                                           'rtsp://admin:ncbc1234@192.168.1.157', 
                                           'rtsp://admin:ncbc1234@192.168.1.158',
                                           'rtsp://admin:ncbc1234@192.168.1.159',
                                           'rtsp://admin:ncbc1234@192.168.1.160',
                                           'rtsp://admin:ncbc1234@192.168.1.161',
                                           'rtsp://admin:ncbc1234@192.168.1.162'))

        if gpu_option == 'CPU':
            model = custom1(path_or_model=path_model_file, gpu=False)
            
        if gpu_option == 'GPU':
            model = custom2(path_or_model=path_model_file, gpu=True)
            if torch.cuda.is_available():
                device =torch.device("cuda:0")
                print("Running on the GPU")

        cap = cv2.VideoCapture(f'{rtsp_url}')
        stframe1 = st.empty()
        while True:
            success, img = cap.read()
            if not success:
                st.error(
                f'RSTP channel NOT working\nChange channel or Connect properly!!',
                icon="🚨"
                    )
                break

            bbox_list = []
            current_no_class = []
            results = model(img)
                
                # Bounding Box
            box = results.pandas().xyxy[0]
            class_list = box['class'].to_list()

            for i in box.index:
                xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                    int(box['ymax'][i]), box['confidence'][i]
                if conf > confidence:
                    bbox_list.append([xmin, ymin, xmax, ymax])
            if len(bbox_list) != 0:
                for bbox, id in zip(bbox_list, class_list):
                    plot_one_box(bbox, img, label=class_labels[id],
                                color=color_pick_list[id], line_thickness=draw_thick)
                    current_no_class.append([class_labels[id]])
            FRAME_WINDOW.image(img)


                # FPS
            oldfps=math.ceil(fps/oldfps)
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
                

            with stframe1.container():
                col1, col2,col3 = st.columns(3)

                col1.metric("Frames Per Second", round(fps), oldfps)
                col2.metric("Camera IP",rtsp_url[22:])
                col3.metric("Screen Resolution",'1280x720')
                

