import streamlit as st
from multiapp import MultiApp
import demo

app = MultiApp()
st.set_page_config(page_title="AttnGAN", initial_sidebar_state="auto")

st.sidebar.title("Navigation")
# Add all application here

app.add_app("Demo", demo.demo_gan)


# The main app
app.run()
