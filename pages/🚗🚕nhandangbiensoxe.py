import requests, os, sys
import json
import streamlit as st
from PIL import Image

st.title('🚗🚕Nhan dang bien so xe🚗🚕')
st.write('https://github.com/NanoNets/nanonets-ocr-sample-python')
image_file = st.file_uploader('Open Image', type=['PNG','JPG','BMP'])
col1, col2 = st.columns([5,2])
if image_file is not None:
    file_name = image_file.name
    image_path = "./images/" + file_name
    image = Image.open(image_path)
    with col1:
        st.image(image)
    model_id = '06daf2be-8412-4e7c-a4e5-7c3aec7679b2'
    api_key = 'OkRErMwrDab5PC6C2UX3gRB0UZtyfKQa'

    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'

    data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}

    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)

    model = json.loads(response.text)["result"][0]["prediction"][0]["ocr_text"]
    if st.button("Nhan dang"):
        with col2:
            st.markdown("### Bien so xe :")
            st.write(model)
            
