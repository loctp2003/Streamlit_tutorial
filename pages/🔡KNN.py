import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import joblib
import cv2
from PIL import ImageTk, Image
import io
st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.title('KNN')
def Bai01():
    st.header('Bai01 📣')
    N = 150
    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(N, centers=np.array(centers), random_state=1)
    nhom_0 = []
    nhom_1 = []
    nhom_2 = []
    for i in range(150):
        if labels[i] == 0:
            nhom_0.append([data[i, 0], data[i, 1], "red", 0])
        elif labels[i] == 1:
            nhom_1.append([data[i, 0], data[i, 1], "green", 1])
        else:
            nhom_2.append([data[i, 0], data[i, 1], "blue", 2])
    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)

    df = pd.DataFrame((*nhom_0, *nhom_1, *nhom_2), columns=["x", "y", "color", "nhom"])
    st.expander("Show data").write(df)

    c = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("color", scale=None),
            tooltip=["x", "y", "color", "nhom"],
        )
    )
    st.altair_chart(c, use_container_width=True)
    base = alt.Chart(df).encode(alt.X('X:O'))
    chart_test_count = base.mark_line().encode(alt.Y('Y:N'))
    chart_test_failures = base.mark_line().encode(alt.Y('Color:N'))
    res = train_test_split(data, labels, 
                            train_size=0.8,
                            test_size=0.2,
                            random_state=12)
    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels) 
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.latex('Do chinh xac: %.0f%%' % (accuracy*100))
def Bai02():
    st.header('Bai02 📣')
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

    # 784 = 28x28
    RESHAPED = 784
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED) 

    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
        test_size=0.1, random_state=84)
    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    # save model, sau này ta sẽ load model để dùng
    def pickle_model(model):
        f = io.BytesIO()
        joblib.dump(model, f)
        return f 
    # Đánh giá trên tập validation
    predicted = model.predict(valData)
    do_chinh_xac = accuracy_score(valLabels, predicted)
    st.write('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))
    # Đánh giá trên tập test
    predicted = model.predict(X_test)
    do_chinh_xac = accuracy_score(Y_test, predicted)
    st.write('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))
    st.download_button("Download Model", data=pickle_model(model), file_name="knn_mnist.pkl")
def Bai03():
    st.header("Bai03 📣")
    uploaded_file = st.file_uploader("OPEN MODEL",type=['pkl'])
    if uploaded_file is not None:
        mnist = keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        index = None
        knn = joblib.load(uploaded_file)
        btn1 = st.button('Nhan dang')
        if btn1:
            col1, col2 = st.columns([3,2])
            index = np.random.randint(0, 9999, 100)
            digit = np.zeros((10*28,10*28), np.uint8)
            k = 0
            for x in range(0, 10):
                for y in range(0, 10):
                        digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                        k = k + 1  
            with col1:
                st.latex("IMAGE")
                st.write()
                st.write()
                cv2.imwrite('digit.jpg', digit)
                image = Image.open('digit.jpg')
                st.image(image, caption='IMAGE')
                sample = np.zeros((100,28,28), np.uint8)
                for i in range(0, 100):
                    sample[i] = X_test[index[i]]
                    
                RESHAPED = 784
                sample = sample.reshape(100, RESHAPED) 
                predicted = knn.predict(sample)
                k = 0
                with col2:
                    st.latex("Ket qua nhan dang")
                    for x in range(0, 10):
                        ketqua = ''
                        for y in range(0, 10):
                            ketqua = ketqua + '%3d' % (predicted[k])
                            k = k + 1
                        st.subheader(ketqua )
                   
page = st.sidebar.selectbox('Select page',['Bai01','Bai02','Bai03']) 
if page == 'Bai01':
    Bai01()
elif page == 'Bai02':
    Bai02()
else :
    Bai03()