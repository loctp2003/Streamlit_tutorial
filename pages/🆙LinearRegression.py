import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import streamlit as st
st.title("🆙LINEAR REGRESSION")
def Bai01():
    st.title('Bai01')
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    mot = np.ones((1, 13), dtype = np.int32)
    X_bar = np.vstack((mot, X))
    X_bar_T = X_bar.T
    A = np.matmul(X_bar, X_bar_T)
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    b = np.matmul(X_bar, y)
    A_inv = np.linalg.pinv(A)
    w = np.matmul(A_inv, b)
    st.markdown("### Chieu cao 🚶")
    st.write(X)
    st.markdown("### Can nang 🫄")
    st.write(y.T)
    col1, col2 = st.columns([3, 1])
    with col1:
        x1 = X[0, 0]
        y1 = x1*w[1, 0] + w[0, 0]
        x2 = X[0, -1]
        y2 = x2*w[1, 0] + w[0, 0]
        plt.plot(X, y.T, 'ro')
        plt.plot([x1, x2], [y1, y2])
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    with col2:
        st.write("w :",w)
        #Tinh sai so
        loss = 0
        sai_so = 0;
        for i in range(0, 13):
            y_mu = np.matmul(X_bar[:,i],w)
            sai_so = y[i, 0] - y_mu
            loss = loss + sai_so**2
        loss = loss/(2*13)
        st.write('Sai so :',loss)
        #Tinh sai so ham norm
        loss = np.linalg.norm(y - np.matmul(X_bar_T, w))
        loss = loss**2
        loss = loss/(2*13)
        st.write('Sai so:(ham norm) ')
        st.write(loss)
    sample = st.sidebar.number_input('Nhap chieu cao 🚶:')
    btn_giai = st.sidebar.button('Tinh')
    if btn_giai not in st.session_state:
        st.session_state.btn_giai = False
    if btn_giai or st.session_state.btn_giai: 
        st.session_state.btn_giai = True
        ket_qua = sample*w[1, 0] + w[0, 0]
        st.sidebar.write('Chieu cao 🚶: %d thi can nang 🫄 la: %d' % (sample, ket_qua))
    
        
        
def Bai02():
    st.title('Bai02 👏')
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    st.markdown("### Chieu cao 🚶")
    st.write(X.T)
    st.markdown("### Can nang 🫄")
    st.write(y.T)
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    st.subheader("scikit-learn’s solution :")
    st.write("w_1= ",regr.coef_[0])
    st.write( "w_0 = ", regr.intercept_)
    y_mu = regr.predict(X)
    loss = mean_squared_error(y, y_mu)
    loss = loss/2
    st.write("Sai so 🔢 :",loss)

def Bai03():
    st.title('Bai03 👏')
    X = np.array([[ 0,  1],
       [ 5,  1],
       [15,  2],
       [25,  5],
       [35, 11],
       [45, 15],
       [55, 34],
       [60, 35]])
    y = np.array([[4, 5, 20, 14, 32, 22, 38, 43]]).T
    st.write("X:",X.T)
    st.write("y:",y.T)
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    st.write("regr.coef_ :",regr.coef_)
    st.write("regr.intercept_ :",regr.intercept_)
    y_mu = regr.predict(X)
    loss = mean_squared_error(y, y_mu)
    loss = loss/2
    st.write('Sai so 🔢:',loss)
def Bai04():
    st.title('Bai04 👏')
    col1,col2 = st.columns(2)
    X = np.array([[35.3, 29.7, 30.8, 58.8, 61.4, 71.3, 74.4, 76.7, 70.7, 57.5, 46.4, 28.9, 28.1, 39.1, 46.8, 48.5, 59.3, 70.0, 70.0, 74.5, 72.1, 58.1, 44.6, 33.4, 28.6],
              [20, 20, 23, 20, 21, 22, 11, 23, 21, 20, 20, 21, 21, 19, 23, 20, 22, 22, 11, 23, 20, 21, 20, 20, 22]]).T

    y = np.array([10.98, 11.13, 12.51, 8.40, 9.27, 8.73, 6.36, 8.50, 7.82, 9.14, 8.24, 12.19, 11.88, 9.57, 10.94, 9.58, 10.09, 8.11, 6.83, 8.88, 7.68, 8.47, 8.86, 10.36, 11.08])
    with col1:
        st.markdown("### X")
        st.write(X)
    with col2:
        st.markdown("### y")
        st.write(y)
    regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    w1 = regr.coef_[0]
    w2 = regr.coef_[1]
    b = regr.intercept_
    st.write('1. b  = %.4f' % b)
    st.write('2. w1 = %.4f' % w1)
    st.write('3. w2 = %.4f' % w2)

    y_mu = regr.predict(X)
    st.write("4. y_mu :",y_mu.T)
    sai_so = mean_squared_error(y,y_mu)
    N = X.shape[0]
    st.write("Sai so 🔢:",sai_so*N)
page = st.sidebar.selectbox('Select page',['Bai01','Bai02','Bai03','Bai04'])
if page == 'Bai01':
    Bai01()
elif page == 'Bai02':
    Bai02()
elif page == 'Bai03':
    Bai03()
else :
    Bai04()


#loss (hàm tổn thất tính sai số)
