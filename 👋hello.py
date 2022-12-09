import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Project cuối kì môn Học máy! 👋")
st.markdown(
    """
    Link :
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Thành viên:
    - Vũ Đức Lộc   21110535
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)