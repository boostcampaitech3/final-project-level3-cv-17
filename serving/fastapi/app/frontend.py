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

# root_password = 'password'

@st.cache
def dehazing(uploaded_file, image_bytes): # user upload image
    files = [
        ('files', (uploaded_file.name, image_bytes,
                    uploaded_file.type))
    ]
    response = requests.post("http://localhost:30001/dehazing", files=files)
    dehaze_image_bytes = response.content 
    dehaze_image = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    return dehaze_image_bytes, dehaze_image # dehazed image

@st.cache
def segmentation(image_bytes): # dehazed image
    files = [
        ('files', (image_bytes))
    ]
    response = requests.post("http://localhost:30001/segmentor", files=files)
    segment_bytes = response.content 
    segment_image = Image.open(io.BytesIO(response.content)).convert('L') # 'L' Grayscale
    
    return segment_bytes, segment_image # segment sky image

@st.cache
def replacement(dehaze_image_bytes, mask_image_bytes, sky_image_bytes): # dehazed image, segment sky image, sky image
    files = [
        ('files', (dehaze_image_bytes)),
        ('files', (mask_image_bytes)),
        ('files', (sky_image_bytes))
    ]
    response = requests.post("http://localhost:30001/replace", files=files)
    final = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    return final


def main():
    st.title("Dehazing Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        with st.container():
            col1, col2, col3 = st.columns([2,1,2])

            with col1:
                st.header("Input Image")
                st.image(image, caption='Uploaded Image')

            with col2:
                st.header("원하는 구름 이미지")
                with st.expander("원하는 구름 이미지를 선택해주세요"):
                    option = st.selectbox(
                        ' ',
                        ('선택 안 함', '작은 구름', '큰 구름', '분홍 구름'))

                    # DB에서 불러온 sky image - 서버 경로 지정
                    st.image("/opt/ml/input/data/Sky/sky001.jpeg", caption = '작은 구름')
                    st.image("/opt/ml/input/data/Sky/sky002.jpg", caption = '큰 구름')
                    st.image("/opt/ml/input/data/Sky/sky003.jpg", caption = '분홍 구름')
                    
                    ### 사용자가 구름 사진 고르고 있는 중..
                    ##### DEHAZING MODEL #####
                    dehaze_image_bytes, dehaze_image = dehazing(uploaded_file, image_bytes)
                    
            
            ### 원하는 구름 누를 때마다 옆에 이미지가 바뀌도록 해도 괜찮을 듯
            ### get_prediction을 나중에 백그라운드 실행으로 하면 될 듯 -> 저게 실행될 동안 다른 UI가 화면에 나타나질 않음

            with col3:
                st.header("Dehazed Image")
                st.spinner("dehazing now...")
                st.image(dehaze_image, caption='Dehazed 이미지')
                ##### SKY SEGMENTATION MODEL #####
                segment_bytes, segment = segmentation(dehaze_image_bytes)

                if option == '선택 안 함':
                    pass
                elif option == '작은 구름':
                    image = Image.open("/opt/ml/input/data/Sky/sky001.jpeg")
                elif option == '큰 구름':
                    image = Image.open("/opt/ml/input/data/Sky/sky002.jpg")
                elif option == '분홍 구름':
                    image = Image.open("/opt/ml/input/data/Sky/sky003.jpg")

                # 선택된 sky image bytes
                sky_byte_arr = io.BytesIO()
                image.save(sky_byte_arr, format='PNG')
                sky_byte_arr = sky_byte_arr.getvalue()
                ##### SKY REPLACEMENT #####
                sky_replace = replacement(dehaze_image_bytes, segment_bytes, sky_byte_arr)
                st.image(sky_replace, caption='Sky Replacement 이미지')

            # 이미지 저장
            buf = io.BytesIO()
            sky_replace.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label='Download Image',
                data=byte_im,
                file_name='dehazed_image.jpg',
                mime='image/jpg'
            )

    # 처음에 나오는 예시 사진
    # 이미지를 업로드하면 사라지게 된다.
    else:
        with st.container():
            st.header('예시 사진')
            col0, col1, col2, col3 = st.columns([1,2,2,1])
            with col1:
                st.header("Dehazing 원하는 사진")
                st.image("/opt/ml/input/data/Hidden/hazy/001_baek.jpg")
            with col2:
                st.header("Dehazing 완료 사진")
                st.image("/opt/ml/input/data/Hidden/gt/001_baek.jpeg")


# @cache_on_button_press('Authenticate')
# def authenticate(password) -> bool:
#     return password == root_password

# password = st.text_input('password', type="password")

# if authenticate(password):
#     st.success('You are authenticated!')
#     main()
# else:
#     st.error('The password is invalid.')

main()
