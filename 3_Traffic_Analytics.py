import streamlit as st
import pandas as pd
import plost
import numpy as np
import altair as alt
import cv2
import torch
from utils.hubconf import custom1,custom2
from utils.plots import plot_one_box
import tempfile
from PIL import ImageColor
import datetime
from collections import Counter
import json
import psutil
import subprocess
import math
import time
import plotly.graph_objects as go
import random
from statistics import mean

p_time=0
traffic=[15,30]
ctraffic=[5,10]

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: White;'>Traffic Analytics</h1>", unsafe_allow_html=True)
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
            'GPU is Available on this Device, Choose GPU for the best performance',icon="✅")
    # Confidence
    confidence = 0.2
    # Draw thickness
    class_labels = path_to_class_txt

    # RTSP
    if options == 'RTSP':

        rtsp_url =st.sidebar.selectbox('Camera IP',
                                           ('rtsp://admin:',))
        if gpu_option == 'CPU':
            model = custom1(path_or_model=path_model_file, gpu=False)
        if gpu_option == 'GPU':
            model = custom2(path_or_model=path_model_file, gpu=True)
            if torch.cuda.is_available():
                device =torch.device("cuda:0")
                print("Running on the GPU")
        cap = cv2.VideoCapture(f'{rtsp_url}')
        stframe1 = st.empty()
        stframe2 =st.empty()
        stframe3 =st.empty()
        stframe4 =st.empty()
        stframe5 =st.empty()
        while True:
            success, img = cap.read()
            if not success:
                st.error(f'RSTP channel NOT working\nChange channel or Connect properly!!',icon="🚨")
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
                    plot_one_box(bbox, img, label=class_labels[id])
                    current_no_class.append([class_labels[id]])
                # FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time  
            with stframe1.container():
                col1, col2,col3 = st.columns(3)
                col1.metric("Data Generation Per Second", round(fps*3))
                col2.metric("Camera IP",rtsp_url[22:])
                col3.metric("Screen Resolution",'1280x720')

            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
            class_fq = json.dumps(class_fq, indent = 4)
            class_fq = json.loads(class_fq)
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
            date=str(datetime.datetime.now())
            dict2={'Day':[],"Time":[],"Date":[],"Source":[]}
            p=str(df_fq.shape)[1]
            for i in range (int(p)):
                dict2["Date"].append(date[:10])
                dict2["Time"].append(date[11:16])
                dict2["Day"].append(datetime.datetime.now().strftime('%A'))
                dict2["Source"].append(rtsp_url[22:])
                df=pd.DataFrame.from_dict(dict2)
            df_fq =pd.concat([df_fq,df],axis=1,join='inner')
            V=pd.read_csv('VMR')
            V=V.append(df_fq)
            V.to_csv('VMR',index=False)   
            vmr=pd.read_csv('VMR')
            with stframe2.container():
                st.markdown("<h2 style='text-align: center; color: White;'>Traffic Distribution</h2>", unsafe_allow_html=True)
                labels = 'Bus', 'Car', 'Motor Cycle', 'Person','Rickshaw','Truck'
                sizes=[]
                for lab in (labels):
                    sizes.append(list(vmr.loc[(vmr.Class ==lab )].sum())[1])
                fig_target = go.Figure(data=[go.Pie(labels=labels,values=sizes,hole=.35)])
                st.plotly_chart(fig_target, use_container_width=True)

            with stframe3.container():
                col4, col5,col6 = st.columns(3)
                a= [random.randrange(0,5,1) for i in range (10)],[random.randrange(1,10,1) for i in range (10)], [random.randrange(1,6,1) for i in range (10)],[random.randrange(1,8,1) for i in range (10)], [random.randrange(1,6,1) for i in range (10)],[random.randrange(1,5,1) for i in range (10)]
                df=pd.DataFrame(a).transpose().rename(columns={0:'Bus',1:'Car',2:'Motor Cycle',3:'Person',4:'Rickshaw',5:'Truck'})
                cc1,cc2=st.columns([1,1])
                with cc1:
                    st.line_chart(df)
                with cc2:
                    st.area_chart(df)
            with stframe4.container():
                col4, col5,col6 = st.columns(3)
                
                col4.metric("Cuumulative Traffic", math.ceil(traffic[-1]), math.ceil(((traffic[-1]-traffic[-2])/traffic[-2])*100))
                col5.metric("Curent Traffic", ctraffic[-1],math.ceil(((ctraffic[-1]-ctraffic[-2])/ctraffic[-2])*100))
                col6.metric("Average Traffic",round(sum(ctraffic) / len(ctraffic))) 
            
            with stframe5.container():
                c1,c2=st.columns((9,1))
                with c1:
                    st.markdown("<h2 style='text-align: center; color: White;'>Cumulative Analysis </h2>", unsafe_allow_html=True)
                    
                    vmr.dropna(inplace=True)
                    vmr['Number'] = vmr['Number'].astype(float)
                    vmr1=vmr.tail(10)
                    j=vmr1.groupby(['Class']).sum().sum()
                    ctraffic.append(j//4)  
                    traffic.append(math.ceil(j//4+(traffic[-1])))
                    chart_data = pd.DataFrame(traffic,columns=['Cummulative Traffic'])
                    st.line_chart(chart_data)

                 

                

