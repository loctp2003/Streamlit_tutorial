import sys
import streamlit as st
import numpy as np
from PIL import  Image
import os.path
import cv2
import joblib
from sklearn.svm import LinearSVC
st.title("Nhan dang khuan mat👩🧑‍🦰👨‍🦰")
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)
detector.setInputSize((320, 320))
recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec.onnx","")
svc = joblib.load('svc.pkl')
mydict = ['BanNinh','BanThanh','ThayDuc']
image_file = st.file_uploader('Open Image 🖼', type=['PNG','JPG','BMP'])
if image_file is not None:
    file_name = image_file.name
    file_name = "test/" + file_name
    st.write(file_name)
    col1, col2 = st.columns(2)
    image = Image.open(image_file)
    with col1:
        st.image(image)
    if st.button('Nhan Dang'):
        with col2:
            img_array = cv2.imread(file_name)
            faces = detector.detect(img_array)
            face_align = recognizer.alignCrop(img_array, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_prediction = svc.predict(face_feature)
            result = mydict[test_prediction[0]]
            cv2.putText(img_array,result,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.imwrite('digit1.jpg', img_array)
            imge1 = Image.open('digit1.jpg')
            st.image(imge1, caption='')

