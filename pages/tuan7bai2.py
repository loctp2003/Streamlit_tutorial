import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
a=st.select_slider('X', [0,1,2,3,4,5,6,7,8,9,10,11])
b=st.select_slider('Y', [0,1,2,3,4,5,6,7,8,9,10,11])
x = np.linspace(-2, 2, 11)
y = np.linspace(-2, 2, 11)
X, Y = np.meshgrid(x,y)
Z = X**a + Y **b
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title='IRR', autosize=False,
                  margin=dict(l=40, r=40, b=40, t=40))
st.plotly_chart(fig)
ax = plt.axes(projection="3d")
