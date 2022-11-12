import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000)
    plt.plot(X, y, 'bo', markersize = 2)
    
    # Chuyen mang 1 chieu thanh vector
    X = np.array([X])
    y = np.array([y])
    # Chuyen vi ma tran
    X = X.T
    y = y.T
    model = LinearRegression()
    model.fit(X, y)
    w0 = model.intercept_
    w1 = model.coef_[0]
    x0 = 0
    y0 = w1*x0 + w0
    x1 = 1
    y1 = w1*x1 + w0
    plt.plot([x0, x1], [y0, y1], 'r')
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    
if __name__ == '__main__':
    main()