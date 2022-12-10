import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Plotting", page_icon="📈")
st.title("📉Giam dan dao ham📈")
def Bai01():
    st.markdown("# Bai01 💹")
    a=st.sidebar.select_slider('Pick a mark', [0,1,2,3,4,5,6,7,8,9,10,11])
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
        st.subheader('x = %4f,cost = %.4f va so lan lap = %d' % (x1[-1], cost(x1[-1]), it1)) 
        x = np.linspace(-6, 6, 100)
        y = x**2 + 5*np.sin(x)
        c = (
        alt.Chart(pd.DataFrame({"x":x,"y":y,"color":"blue"}))
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("color", scale=None),
            tooltip=["x", "y", "color"],
        )
    )
        X = x1[b]
        Y = cost(x1[b])
        lines = (
        alt.Chart(pd.DataFrame({"x": [X], "y": [Y],"color":"red"}))
        .mark_circle(size=400)
        .encode(x="x:Q", y="y:Q",color = alt.Color("color", scale=None),tooltip=["x", "y", "color"],))
        st.altair_chart(c+lines, use_container_width=True)
        s = 'iter %d/%d, grad = %.4f' % (b, it1, grad(x1[b]))
        st.sidebar.write(s)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    if __name__ == '__main__':
        main()
def Bai02():
    st.markdown("# Bai02 📉")
    a=st.sidebar.slider('X', 0,50)
    b=st.sidebar.slider('Y', 0,50)
    st.latex("Z=X^{%d}+Y^{%d}" %(a,b))
    x = np.linspace(-2, 2, 11)
    y = np.linspace(-2, 2, 11)
    X, Y = np.meshgrid(x,y)
    Z = X**a + Y **b
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='IRR', autosize=False,
                    margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)
    ax = plt.axes(projection="3d")
def Bai03():
    st.markdown("# Bai03 📉")
    def main():
        X = np.random.rand(1000)
        y = 4 + 3 * X + .5*np.random.randn(1000)
        z = "blue"
        st.expander("Show data").write(pd.DataFrame({"x":X,"y":y,"z":z}))
        c = (
            alt.Chart(pd.DataFrame({"x":X,"y":y,"color":z}))
            .mark_circle()
            .encode(
                x="x:Q",
                y="y:Q",
                color = alt.Color("color", scale=None),
                tooltip=["x", "y", "color"]
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
        rx = np.array([[0., 1.]]).T
        ry = model.predict(rx)
        lines = (
        alt.Chart(pd.DataFrame({"x": rx.squeeze(), "y": ry.squeeze(),"color":"red"}))
        .mark_line()
        .encode(x="x", y="y",color = alt.Color("color", scale=None))
        )
        
        st.altair_chart(c+lines, use_container_width=True)
        
    if __name__ == '__main__':
        main()
def Bai04():
    st.write("📊")
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    print('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F
    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    temp1 = w1[0]
    bb1 = temp1[0]
    ww1 = temp1[1]
    zz1 = cost(temp1) 
    ax = plt.axes(projection="3d")

    #fig1 = go.Figure(data=[go.Icicle(bb, ww, zz)])
    temp2 = w1[1]
    bb2 = temp2[0]
    ww2 = temp2[1]
    zz2 = cost(temp2) 

    #fig2 = go.Figure(data=[go.Icicle(bb, ww, zz)])
    temp3 = w1[2]
    bb3 = temp3[0]
    ww3 = temp3[1]
    zz3 = cost(temp3) 

    #fig3 = go.Figure(data=[go.Icicle(bb, ww, zz)])
    temp4 = w1[3]
    bb4 = temp4[0]
    ww4 = temp4[1]
    zz4 = cost(temp4) 
    X =[bb1,bb2,bb3,bb4]
    Y =[ww1,ww2,ww3,ww4]
    Z =[zz1,zz2,zz3,zz4]

    fig = px.scatter_3d(x= X, y= Y, z = Z)

    fig.add_traces(data=[go.Surface(z=z, x=b, y=w)])
    fig.update_layout(title='IRR', autosize=False,
                    margin=dict(l=40, r=40, b=40, t=40))
    col1, col2 = st.columns([3, 1])
    col1.subheader("A wide column with a chart")
    col1.plotly_chart(fig)
    with col2:
        col2.subheader("A narrow column with the data")
        st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
        dem = 1;
        for item in w1:
            st.write(item, cost(item))
        st.write(len(w1))
page = st.sidebar.selectbox('Select page',['Bai01','Bai02','Bai03','Bai04']) 
if page == 'Bai01':
    Bai01()
elif page == 'Bai02':
    Bai02() 
elif page == 'Bai03':
    Bai03()
else :
    Bai04()
