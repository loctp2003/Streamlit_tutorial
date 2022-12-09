from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
st.title('SVM 📈')
def Bai01():
    st.header("Bai01 👏")
    N = 150
    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(N, 
                            centers=np.array(centers),
                            random_state=1)
    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=12)
    train_data, test_data, train_labels, test_labels = res 

    svc = SVC(C = 100, kernel='linear', random_state=42)

    svc.fit(train_data, train_labels) 
    predicted = svc.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)

    w = svc.coef_[0]
    intercept = svc.intercept_[0]
    a = -w[0] / w[1]

    nhom_0 = []
    nhom_1 = []
    for i in range(150):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1], "green",0])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]," red",1])
    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    df = pd.DataFrame((*nhom_0, *nhom_1), columns=["x", "y", "color","nhom"])
    st.expander("Show data").write(df)

    c = (
        alt.Chart(df)
        .mark_circle(size=70)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("color", scale=None),
            tooltip=["x", "y", "color","nhom"],
        )
    )
    xx = np.linspace(2, 7)
    yy = a * xx - intercept / w[1]

    margin = 1 / np.sqrt(np.sum(svc.coef_**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    line1 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy}))
        .mark_line()
        .encode(x="x:Q", y="y:Q")
    )
    line2 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy_down}))
        .mark_line(strokeDash=[4, 2])
        .encode(x="x:Q", y="y:Q")
    )
    line3 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy_up}))
        .mark_line(strokeDash=[4, 2])
        .encode(x="x:Q", y="y:Q")
    )
    b = (
        alt.Chart(pd.DataFrame({"x":svc.support_vectors_[:, 0] , "y":svc.support_vectors_[:, 1],"z":"blue"}))
        .mark_square(size=70)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("z", scale=None),
        )
    )
    support_vectors = svc.support_vectors_
    st.altair_chart(c+line1+line2+line3+b, use_container_width=True)
    st.write('Do chinh xac 🙌 : %.0f%%' % (accuracy*100))
    sai_so = accuracy_score(test_labels, predicted)
    st.write('sai so 🙌:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)
    st.write('Ket qua nhan dang la nhom :', ket_qua[0])
    st.write(support_vectors)
    base = alt.Chart(df).encode(alt.X('X:O'))
    chart_test_count = base.mark_line().encode(alt.Y('Y'))
    chart_test_failures = base.mark_line().encode(alt.Y('Color:N'))
def Bai02():
    st.header("Bai02 👏")
    N = 150
    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(N, 
                            centers=np.array(centers),
                            random_state=1)
    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=12)
    train_data, test_data, train_labels, test_labels = res 

    svc = LinearSVC(max_iter = 10000)

    svc.fit(train_data, train_labels) 
    predicted = svc.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)

    w = svc.coef_[0]
    intercept = svc.intercept_[0]
    a = -w[0] / w[1]

    nhom_0 = []
    nhom_1 = []
    for i in range(150):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1], "green",0])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1],"red",1])
    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    df = pd.DataFrame((*nhom_0, *nhom_1), columns=["x", "y", "color","nhom"])
    st.expander("Show data").write(df)
    c = (
        alt.Chart(df)
        .mark_circle(size=70)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("color", scale=None),
            tooltip=["x", "y", "color","nhom"],
        )
    )

    xx = np.linspace(2, 7)
    yy = a * xx - intercept / w[1]
    line1 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy}))
        .mark_line()
        .encode(x="x:Q", y="y:Q")
    )
    xx = np.linspace(3, 8)                                                  
    yy = a * xx - intercept / w[1]  + 0.5 / w[1]
    line2 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy}))
        .mark_line(strokeDash=[4, 2])
        .encode(x="x:Q", y="y:Q")
    )
    xx = np.linspace(2, 6)                                                  
    yy = a * xx - intercept / w[1]  - 0.5 / w[1]
    plt.plot(xx, yy, 'b--')
    line3 = (
        alt.Chart(pd.DataFrame({"x": xx, "y": yy}))
        .mark_line(strokeDash=[4, 2])
        .encode(x="x:Q", y="y:Q")
    )
    plt.legend([0,1])
    st.altair_chart(c+line1+line2+line3, use_container_width=True)
    st.latex('Do chinh xac: %.0f%%' % (accuracy*100))
page = st.sidebar.selectbox('Select page',['Bai01','Bai02'])
if page == 'Bai01':
    Bai01()
elif page == 'Bai02':
    Bai02()


