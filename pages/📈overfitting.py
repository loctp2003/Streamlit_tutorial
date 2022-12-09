import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import streamlit as st
import pandas as pd
import altair as alt
st.set_page_config(page_title="Plotting", page_icon="📈")
def Bai01():
    st.header("Bai01 📉")
    N = 30
    N_test = 20 
    def main():
        #khởi tạo bộ tạo số ngẫu nhiên
        np.random.seed(100)
        X_true = np.linspace(0, 5, 51)
        y_true = 3*(X_true -2) * (X_true - 3)*(X_true-4)
        X = np.random.rand(N, 1)*5
        
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        z = "red"
        c = (
            alt.Chart(pd.DataFrame({"x":X[:,0],"y":y[:,0],"color":z}))
            .mark_circle()
            .encode(
                x="x:Q",
                y="y:Q",
                color = alt.Color("color", scale=None)
                
            )
        )
        z2="yellow"
        line1 = (
        alt.Chart(pd.DataFrame({"x": X_true, "y": y_true,"color":z2}))
        .mark_line()
        .encode(x="x:Q", y="y:Q",color = alt.Color("color", scale=None),)
        )
        
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        
        w0 = lin_reg.intercept_[0]
        w1 = lin_reg.coef_[0,0]
        w2 = lin_reg.coef_[0,1]
        
        y_predict = w0 + X_true*w1 + X_true**2*w2
        z3="blue"
        line2 = (
        alt.Chart(pd.DataFrame({"x": X_true, "y": y_predict,"z":z3}))
        .mark_line()
        .encode(x="x:Q", y="y:Q",color = alt.Color("z", scale=None),)
        )

        #tinh sai so tren tap test
        X_test_poly = poly_features.fit_transform(X_test)
        y_test_predict = lin_reg.predict(X_test_poly)
        mse_test = mean_squared_error(y_test, y_test_predict)
        rmse_test = np.sqrt(mse_test)    
        st.altair_chart(c+line1+line2, use_container_width=True)
        st.write(lin_reg.intercept_, lin_reg.coef_)   
        st.write('Sai so binh phuong trung binh - test: %.4f' % rmse_test)

        
    if __name__ == '__main__':
        main()
def Bai02(): 
    st.header("Bai02📉")     
    N = 30
    N_test = 20 
    def main():
        #khởi tạo bộ tạo số ngẫu nhiên
        np.random.seed(100)
        
        X_true = np.linspace(0, 5, 51)
        y_true = 3*(X_true -2) * (X_true - 3)*(X_true-4)
        
        X = np.random.rand(N, 1)*5
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        c = (
            alt.Chart(pd.DataFrame({"x":X[:,0],"y":y[:,0],"color":"red"}))
            .mark_circle()
            .encode(
                x="x:Q",
                y="y:Q",
                color = alt.Color("color", scale=None)
                
            )
        )
        z2="yellow"
        line1 = (
        alt.Chart(pd.DataFrame({"x": X_true, "y": y_true,"color":z2}))
        .mark_line()
        .encode(x="x:Q", y="y:Q",color = alt.Color("color", scale=None))
        )
        poly_features = PolynomialFeatures(degree=4, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        
        print(lin_reg.intercept_, lin_reg.coef_)
        w0 = lin_reg.intercept_[0]
        w1 = lin_reg.coef_[0,0]
        w2 = lin_reg.coef_[0,1]
        w3 = lin_reg.coef_[0,2]
        w4 = lin_reg.coef_[0,3]

        y_predict = w0 + X_true*w1 + w2*X_true**2 + w3*X_true**3 + w4*X_true**4
        z3="blue"
        line2 = (
        alt.Chart(pd.DataFrame({"x": X_true, "y": y_predict,"color":z3}))
        .mark_line()
        .encode(x="x:Q", y="y:Q",color = alt.Color("color", scale=None))
        )
        st.altair_chart(c+line1+line2, use_container_width=True)
        #tinh sai so tren tap test
        X_test_poly = poly_features.fit_transform(X_test)
        y_test_predict = lin_reg.predict(X_test_poly)
        mse_test = mean_squared_error(y_test, y_test_predict)
        rmse_test = np.sqrt(mse_test)
        st.write('Sai so binh phuong trung binh - test  : %.4f'% rmse_test)
        
        
    if __name__ == '__main__':
        main()
def Hoiquydathuc():
    def Hoiquybac2():
        st.header("Hoiquybac2 📉")
        np.random.seed(100)
        N = 30
        X = np.random.rand(N, 1)*5
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
        poly_features = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly_features.fit_transform(X)
        N_test = 20 
        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        X_poly_test = poly_features.fit_transform(X_test)
        lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
        lin_reg.fit(X_poly, y)
        x_ve = np.linspace(-2, 10, 100)
        y_ve = np.zeros(100, dtype = np.float64)
        y_real = np.zeros(100, dtype = np.float64)
        x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
        y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
        for i in range(0, 100):
            y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)
        # Tinh sai so cua scikit-learn
        y_train_predict = lin_reg.predict(X_poly)
        # print(y_train_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
        plt.plot(X,y, 'ro')
        plt.plot(X_test,y_test, 's')
        plt.plot(x_ve, y_ve, 'b')
        plt.plot(x_ve, y_real, '--')
        plt.title('Hoi quy da thuc bac 2')
        st.pyplot()
        st.markdown('#### 1. Sai so binh phuong trung binh - tap training  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
        # Tinh sai so cua scikit-learn
        y_test_predict = lin_reg.predict(X_poly_test)
        # print(y_test_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
        st.markdown('#### 2. Sai so binh phuong trung binh - tap test  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
    def Hoiquybac4():
        st.header("Hoiquybac4 📉")
        np.random.seed(100)
        N = 30
        X = np.random.rand(N, 1)*5
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
        poly_features = PolynomialFeatures(degree=4, include_bias=True)
        X_poly = poly_features.fit_transform(X)
        N_test = 20 
        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        X_poly_test = poly_features.fit_transform(X_test)
        lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
        lin_reg.fit(X_poly, y)
        x_ve = np.linspace(-2, 10, 100)
        y_ve = np.zeros(100, dtype = np.float64)
        y_real = np.zeros(100, dtype = np.float64)
        x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
        y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
        for i in range(0, 100):
            y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)
        # Tinh sai so cua scikit-learn
        y_train_predict = lin_reg.predict(X_poly)
        # print(y_train_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
        plt.plot(X,y, 'ro')
        plt.plot(X_test,y_test, 's')
        plt.plot(x_ve, y_ve, 'b')
        plt.plot(x_ve, y_real, '--')
        plt.title('Hoi quy da thuc bac 4')
        st.pyplot()
        st.markdown('#### 1.Sai so binh phuong trung binh - tap training  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
        # Tinh sai so cua scikit-learn
        y_test_predict = lin_reg.predict(X_poly_test)
        # print(y_test_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
        st.markdown('#### 2.sai so binh phuong trung binh - tap test  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
    def Hoiquybac8():
        st.header('Hoiquybac8 📣')
        np.random.seed(100)
        N = 30
        X = np.random.rand(N, 1)*5
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
        poly_features = PolynomialFeatures(degree=8, include_bias=True)
        X_poly = poly_features.fit_transform(X)
        N_test = 20 
        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        X_poly_test = poly_features.fit_transform(X_test)
        lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
        lin_reg.fit(X_poly, y)
        x_ve = np.linspace(-2, 10, 100)
        y_ve = np.zeros(100, dtype = np.float64)
        y_real = np.zeros(100, dtype = np.float64)
        x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
        y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
        # Tinh sai so cua scikit-learn
        y_train_predict = lin_reg.predict(X_poly)
        # print(y_train_predict)
        plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])
        plt.plot(X,y, 'ro')
        plt.plot(X_test,y_test, 's')
        plt.plot(x_ve, y_ve, 'b')
        plt.plot(x_ve, y_real, '--')
        plt.title('Hoi quy da thuc bac 8')
        st.pyplot()
        sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
        for i in range(0, 100):
            y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)
        st.write(np.min(y_test), np.max(y) + 100)
        st.markdown('#### 1.Sai so binh phuong trung binh - tap training  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
        # Tinh sai so cua scikit-learn
        y_test_predict = lin_reg.predict(X_poly_test)
        # print(y_test_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
        st.markdown('#### 2.Sai so binh phuong trung binh - tap test  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
    def Hoiquybac16():
        st.header("Hoiquybac16 📉")
        np.random.seed(100)
        N = 30
        X = np.random.rand(N, 1)*5
        y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)
        poly_features = PolynomialFeatures(degree=8, include_bias=True)
        X_poly = poly_features.fit_transform(X)
        N_test = 20 
        X_test = (np.random.rand(N_test,1) - 1/8) *10
        y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
        X_poly_test = poly_features.fit_transform(X_test)
        lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
        lin_reg.fit(X_poly, y)
        x_ve = np.linspace(-2, 10, 100)
        y_ve = np.zeros(100, dtype = np.float64)
        y_real = np.zeros(100, dtype = np.float64)
        x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
        y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
        plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])
        # Tinh sai so cua scikit-learn
        plt.plot(X,y, 'ro')
        plt.plot(X_test,y_test, 's')
        plt.plot(x_ve, y_ve, 'b')
        plt.plot(x_ve, y_real, '--')
        plt.title('Hoi quy da thuc bac 16')

        st.pyplot()
        for i in range(0, 100):
            y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)
        st.write(np.min(y_test), np.max(y) + 100)
        y_train_predict = lin_reg.predict(X_poly)
        # print(y_train_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
        st.markdown('#### 1.sai so binh phuong trung binh - tap training  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
        # Tinh sai so cua scikit-learn
        y_test_predict = lin_reg.predict(X_poly_test)
        # print(y_test_predict)
        sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
        st.markdown('#### 2.sai so binh phuong trung binh - tap test  : %.6f' % (sai_so_binh_phuong_trung_binh/2))
    btn = st.sidebar.radio("Select",["Hoiquybac2","Hoiquybac4","Hoiquybac8","Hoiquybac16"])
    if btn == "Hoiquybac2":
        Hoiquybac2()
    elif btn == "Hoiquybac4":
        Hoiquybac4()
    elif btn == "Hoiquybac8":
        Hoiquybac8()
    else:
        Hoiquybac16()
            
st.title("OVERFITTING")
page = st.sidebar.selectbox('Select page',['Bai01','Bai02','Hoiquydathuc']) 
if page == 'Bai01':
    Bai01()
elif page == 'Bai02':
    Bai02()
else:
    Hoiquydathuc()