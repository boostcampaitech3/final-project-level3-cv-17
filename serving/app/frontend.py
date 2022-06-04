import io
import os
import time
import numpy as np
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from app.confirm_button_hack import cache_on_button_press


# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

# root_password = 'password'

### front-end 화면 구성 바꾸기
### 2대 2 화면 구성
### 상단은 input 과 dehazed 결과 (2개)
### 하단은 구름 예시 사진들과 구름 합성 최종 사진 (2개)

@st.cache
def dehazing(image_bytes): # user upload image
    files = [
        ('files', (image_bytes))
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
def selection(dehaze_image_bytes, mask_image_bytes, sky_option): # dehazed image, segment sky image
    files = [
        ('files', (dehaze_image_bytes)),
        ('files', (mask_image_bytes)),
    ]
    response = requests.post("http://localhost:30001/select/" + sky_option, files=files)
    sky_paths = response.json()["file_path"]
    return sky_paths

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

def image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG") # JPG, PNG, JPEG
    bytes_img = buf.getvalue()
    return bytes_img

def db_save(cloud_option, files):
    r = requests.post("http://localhost:30001/save/" + cloud_option, files=files)

# 이미지 저장과 save request 분리!
### 이미지 다운로드와 함께 request를 날림 -> app.get('/save') / DB에 저장
def save_btn_click(option, bytes):
    files = [
        ('files', (bytes))
    ]
    ### 이 부분은 일단 프로토타입용으로 작성했음
    ### 나중에 구름 예시 사진들 카테고리가 정리되면 수정하면 됨.
    if option == '선택 안 함':
        cloud_option = 'no_option'
    elif option == '큰 구름':
        cloud_option = 'big'
    elif option == '작은 구름':
        cloud_option = 'small'
    elif option == '분홍 구름':
        cloud_option = 'pink'
    else:
        cloud_option = 'none'

    st.write(f"{cloud_option}으로 다운로드")
    db_save(cloud_option, files)

def test_resize(haze_img, max_size):
    width, height = haze_img.size
    
    if width > max_size or height > max_size:
        if width < height:
            haze_img = haze_img.resize(( int(max_size*(width/height)), max_size ), Image.ANTIALIAS)
        elif width >= height:
            haze_img = haze_img.resize(( max_size, int(max_size*(height/width)) ), Image.ANTIALIAS)
        width, height = haze_img.size
    
    if width%16 != 0 and height%16 != 0:
        haze_img = haze_img.resize((width + 16 - width%16, height + 16 - height%16), Image.ANTIALIAS)
    elif width%16 != 0:
        haze_img = haze_img.resize((width + 16 - width%16, height), Image.ANTIALIAS)
    elif height%16 != 0:
        haze_img = haze_img.resize((width, height + 16 - height%16), Image.ANTIALIAS)
    
    return haze_img

def main():
    # print('main')
    st.title("Dehazing & Sky Replacement")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Image Resize
        image = test_resize(image, 3024)
        image_bytes = image_to_bytes(image)
    
        ### 상단 부분
        with st.container():
            col1, col2, col3 = st.columns([3,1,3])

            with col1:
                st.header("Before Dahazing")
                st.image(image, caption='Uploaded Image')

                ### 현재 사용자가 사진을 업로드한 상태
                ### 이 때, background에서 dehazing inference를 진행하면 된다.
                ##### DEHAZING MODEL #####
                dehaze_image_bytes, dehaze_image = dehazing(image_bytes)
            
            # 공백
            with col2:
                st.write(' ')

            # dehazed inference 결과
            with col3:
                st.header("After Dehazing")
                st.spinner("dehazing now...")
                ### Dehazing 결과를 웹에 출력
                st.image(dehaze_image, caption='Dehazed 이미지')

                
                option = '선택 안 함'
                # dehazed 이미지 저장 버튼
                # 만약 구름 합성 사진이 맘에 안 들면, dehazed된 이미지만 저장할 수 있게 만든 버튼
                down_btn = st.download_button(
                    label='Download Image',
                    data=dehaze_image_bytes,
                    file_name='dehazed_image.jpg',
                    mime='image/jpg',
                    on_click=save_btn_click(option, dehaze_image_bytes)
                )


        ### 하단 부분
        with st.container():
            col1, col2, col3 = st.columns([3,1,3])  # 구름 예시 사진, 빈 공간, 구름 inference 사진

            ### TODO
            ### db에 구름 예시 사진들이 저장되어 있음. (절대 경로로 저장하고 vscode 서버 안에 실제 사진들이 저장)
            ### 이 부분은 CLIP 사용하는 코드가 완성되고, 구름 카테고리도 완성이 되면 수정하면 될 듯
            ### Example))) 분홍 구름 선택 -> CLIP으로 추천 -> db에서 3장 뽑아옴 -> streamlit에 출력
            ### 3개의 사진에 대해 각각 sky replacement를 시작하는 버튼을 추가
            
            ##### SKY SEGMENTATION MODEL #####
            segment_bytes, segment = segmentation(dehaze_image_bytes)
            
            with col1:
                st.header("Choose Cloud Image")
                with st.expander("원하는 구름 이미지를 선택해주세요"):
                    option = st.selectbox(
                        '이미지와 단어에 어울리는 하늘을 추천합니다',
                        ('a pink sky', 'a blue sky', 'a sunset sky', 'a sky with small clouds', 'a sky with large clouds', 'a dark night sky', 'the starry night sky'))

                    selected = selection(dehaze_image_bytes, segment_bytes, option)   

                    if selected:
                        for i, select_sky_path in enumerate(selected):
                            st.image(select_sky_path, width=384)

                        with st.form("Select Cloud"):
                            select_option = st.selectbox(
                            '원하는 하늘을 선택해주세요',
                            [str(i+1)+'번째' for i in range(len(selected))])

                            sky_image = Image.open(selected[int(select_option[0])-1])
                            

                            # 선택된 sky image bytes
                            sky_byte_arr = image_to_bytes(sky_image)

                            submitted = st.form_submit_button('구름 합성 시작')
                            

                if submitted:
                    st.session_state.submitted = True

                ### 해당 버튼을 눌렀을 때, sky replacement가 시작된다.
                ### 추후에 각 이미지(3장) 별로 버튼을 만들면 될 듯.
                # sky_replace = None
                
                if 'submitted' in st.session_state:
                    print(st.session_state.submitted)
                    ##### SKY REPLACEMENT #####
                    sky_replace = replacement(dehaze_image_bytes, segment_bytes, sky_byte_arr)
                    # st.write(sky_replace)

                    # 공백
                    with col2:
                        st.write(' ')

                    
                    with col3:
                        st.header('After Cloud Generate')
                        st.image(segment)
                        ### 위에서 api response를 받은 sky_replace를 여기서 웹에 출력한다.
                        st.image(sky_replace, caption='Sky Replacement 이미지')
                        ### 이미지 다운로드와 함께 request를 날림 -> app.post('/save')
                        ### 사용자가 업로드한 이미지 / 선택한 구름 옵션을 보냄
                        sky_replace_bytes = image_to_bytes(sky_replace)

                        final_down_btn = st.download_button(
                            label='Download Image',
                            data=sky_replace_bytes,
                            file_name='dehazed_cloud_image.jpg',
                            mime='image/jpg',
                            on_click=save_btn_click(option, sky_replace_bytes)
                        )

    ### Challenge : inference하는 과정이 background에서 실행이 되느냐??


    # 웹사이트 접속할 때 처음에 나오는 예시 사진
    # 이미지를 업로드하면 사라지게 된다.
    else:
        with st.container():
            st.header('')
            col0, col1, col2, col3, col4, col5 = st.columns([0.5,2,2,2,2,0.5])
            with col1:
                st.header("Input Image")
                st.image("/opt/ml/input/data/Example/input_image.jpg")
            with col2:
                st.header("After Dehazing")
                st.image("/opt/ml/input/data/Example/dehazed_image.jpg")
            with col3:
                st.header("Sky Selection")
                st.image("/opt/ml/input/data/Example/selected_sky.jpeg")
            with col4:
                st.header("Replacement")
                st.image("/opt/ml/input/data/Example/dehazed_cloud_image.jpg")


main()