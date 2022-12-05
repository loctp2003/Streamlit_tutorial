import numpy as np
import streamlit as st;
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import altair as alt
def main():
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000)
    z = "blue"
    st.expander("Show data").write(pd.DataFrame({"x":X,"y":y,"z":z,"t":2}))
    c = (
        alt.Chart(pd.DataFrame({"x":X,"y":y,"z":z,"t":2}))
        .mark_circle(size=50)
        .encode(
            x="x:Q",
            y="y:Q",
            color = alt.Color("z", scale=None),
        )
    )
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
    
    lines = (
    alt.Chart(pd.DataFrame({"x": [x0, x1], "y": [y0, y1],}))
    .mark_line()
    .encode(x="x", y="y")
    )
    
    st.altair_chart(c+lines, use_container_width=True)
    
if __name__ == '__main__':
    main()