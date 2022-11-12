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

a=st.number_input('Nhap a:')

btn_giai = st.button('Giai')
if btn_giai not in st.session_state:
    st.session_state.btn_giai = False
if btn_giai or st.session_state.btn_giai: 
    def grad(x):
        return 2*x+ 5*np.cos(x)
    def cost(x):
        return x**2 + 5*np.sin(x)
    def myGD1(x0, eta):
        x = [x0]
        for it in range(100):
            x_new = x[-1] - eta*grad(x[-1])
            if abs(grad(x_new)) < 1e-3: 
                break
            x.append(x_new)
        return (x, it)
    def main():
        b=int(a)
        (x1, it1) = myGD1(-5, .1)
        print('x = %4f,cost = %.4f va so lan lap = %d' % (x1[-1], cost(x1[-1]), it1)) 
        x = np.linspace(-6, 6, 100)
        y = x**2 + 5*np.sin(x)
        plt.subplot(2,4,1)
        plt.plot(x, y, 'b')
        plt.plot(x1[b], cost(x1[b]), 'ro')
        s = 'iter %d/%d, grad = %.4f' % (b, it1, grad(x1[b]))
        plt.xlabel(s, fontsize=8)
        st.pyplot()
        plt.tight_layout()
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    if __name__ == '__main__':
        main()