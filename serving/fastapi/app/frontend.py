import io
import os
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from app.confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


@st.cache
def dehazing(uploaded_file, image_bytes):
    files = [
        ('files', (uploaded_file.name, image_bytes,
                    uploaded_file.type))
    ]
    response = requests.post("http://localhost:30/predict", files=files)
    dehaze_image = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    return dehaze_image


def main():
    st.title("Dehazing Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Dehazing...")

        # 기존 stremalit 코드
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]
        dehaze_image = dehazing(uploaded_file, image_bytes)
        st.image(dehaze_image)


@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')
