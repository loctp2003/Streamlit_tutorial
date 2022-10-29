from unittest import result
import streamlit as st
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import plotly.figure_factory as ff

a = st.number_input('Nhap a:')

btn_giai = st.button('Giai')
if "btn_giai_state" not in st.session_state:
    st.session_state.btn_giai = False
if btn_giai or st.session_state.btn_giai_state: 
   

    N = 150
    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []
    nhom_2 = []
    for i in range(150):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1],1,0])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1],2,1])
        else:
            nhom_2.append([data[i,0], data[i,1],3,2])
    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)
    
    df = pd.DataFrame(
        (*nhom_0 , *nhom_1 ,*nhom_2),
        columns=['x','y','color','nhom']
    )
    c = alt.Chart(df).mark_circle().encode(
    x='x' , y='y' , color='color', tooltip=['x', 'y', 'color','nhom'])
    st.altair_chart(c, use_container_width=True)
    base = alt.Chart(df).encode(alt.X('X:O'))
    chart_test_count = base.mark_line().encode(alt.Y('Y:N'))
    chart_test_failures = base.mark_line().encode(alt.Y('Color:N'))
  

    st.set_option('deprecation.showPyplotGlobalUse', False)
    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=12)
    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels) 
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Do chinh xac: %.0f%%' % (accuracy*100))

