import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
st.set_page_config(page_title="Plotting", page_icon="📈")
a=st.select_slider('Pick a mark', [0,1,2,3,4,5,6,7,8,9,10,11])
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
    st.write('x = %4f,cost = %.4f va so lan lap = %d' % (x1[-1], cost(x1[-1]), it1)) 
    x = np.linspace(-6, 6, 100)
    y = x**2 + 5*np.sin(x)
    plt.subplot(2,4,1)
    plt.plot(x, y, 'b')
    plt.plot(x1[b], cost(x1[b]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (b, it1, grad(x1[b]))
    plt.xlabel(s, fontsize=8)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == '__main__':
    main()