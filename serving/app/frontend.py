import io
import os
import time
import numpy as np
from pathlib import Path

import requests
import cv2
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
def check(mask_image_bytes, sky_mask_bytes): # dehazed image, segment sky image, sky image
    files = [
        ('files', (mask_image_bytes)),
        ('files', (sky_mask_bytes))
    ]
    response = requests.post("http://localhost:30001/check", files=files)
    return response.content.decode('utf-8')

@st.cache
def replacement(dehaze_image_bytes, mask_image_bytes, sky_image_bytes, sky_mask_bytes): # dehazed image, segment sky image, sky image
    files = [
        ('files', (dehaze_image_bytes)),
        ('files', (mask_image_bytes)),
        ('files', (sky_image_bytes)),
        ('files', (sky_mask_bytes))
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

    if option == 'a pink sky':
        cloud_option = 'a_pink_sky'
    elif option == 'a blue sky':
        cloud_option = 'a_blue_sky'
    elif option == 'a_sunset_sky':
        cloud_option = 'a_sunset_sky'
    elif option == 'a sky with small clouds':
        cloud_option = 'a_sky_with_small_clouds'
    elif option == 'a sky with large clouds':
        cloud_option = 'a_sky_with_large_clouds'
    elif option == 'a dark night sky':
        cloud_option = 'a_dark_night_sky'
    elif option == 'the_starry_night_sky':
        cloud_option = 'the_starry_night_sky'
    else:
        cloud_option = 'dehazed'

    st.write(f"{cloud_option} 다운로드")
    db_save(cloud_option, files)

def test_resize(haze_img, min_size, max_size, check_size):
    width, height = haze_img.size

    if width < min_size or height < min_size:
        if width < height:
            haze_img = haze_img.resize(( min_size, int(min_size*(height/width)) ), Image.ANTIALIAS)
        elif width >= height:
            haze_img = haze_img.resize(( int(min_size*(width/height)), min_size ), Image.ANTIALIAS)
        width, height = haze_img.size
    
    if width > max_size or height > max_size:
        if width < height:
            haze_img = haze_img.resize(( int(max_size*(width/height)), max_size ), Image.ANTIALIAS)
        elif width >= height:
            haze_img = haze_img.resize(( max_size, int(max_size*(height/width)) ), Image.ANTIALIAS)
        width, height = haze_img.size
    
    if width % check_size != 0 and height % check_size != 0:
        haze_img = haze_img.resize((width - width%check_size, height - height%check_size), Image.ANTIALIAS)
    elif width % check_size != 0:
        haze_img = haze_img.resize((width - width%check_size, height), Image.ANTIALIAS)
    elif height % check_size != 0:
        haze_img = haze_img.resize((width, height - height%check_size), Image.ANTIALIAS)

    return haze_img

def apply_clahe(image):
    clip_limit = 1
    grid_size = 64
    # PIL to CV2 
    numpy_image = np.array(image)  
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    # BGR to HSV
    b,g,r = cv2.split(img)
    img_rgb = cv2.merge([r, g, b])
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size,grid_size))
    img_out = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
    img_out = Image.fromarray(img_out)
    return img_out

def main():
    # print('main')
    st.title("Dehazing & Sky Replacement")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Image Resize
        min_size, max_size = 512, 3024
        check_size = 16
        image = test_resize(image, min_size, max_size, check_size)
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
                # dehaze image CLAHE
                
                dehaze_image_out = apply_clahe(dehaze_image)
                st.image(dehaze_image_out, caption='Dehazed 이미지')

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
            
            ##### SKY SEGMENTATION MODEL #####
            segment_bytes, segment = segmentation(dehaze_image_bytes)

            submitted = False

            with col1:
                st.header("Choose Cloud Image")
                with st.expander("원하는 구름 이미지를 선택해주세요"):
                    option = st.selectbox(
                        """이미지와 단어에 어울리는 하늘을 추천합니다.""",
                        ('Upload sky image', 'a pink sky', 'a blue sky', 'a sunset sky', 'a sky with small clouds', 'a sky with large clouds', 'a dark night sky', 'the starry night sky'))


                    if option == 'Upload sky image':
                        uploaded_sky_file = st.file_uploader("Upload sky image", type=["jpg", "jpeg","png"])

                        if uploaded_sky_file:
                            sky_image_bytes = uploaded_sky_file.getvalue()
                            sky_image = Image.open(io.BytesIO(sky_image_bytes))
                            st.image(sky_image, width=386, caption='업로드한 하늘 이미지', use_column_width='always')
                            
                            # 사용자가 입력한 하늘 이미지에 대한 segment
                            ref_mask_bytes, sky_segment = segmentation(sky_image_bytes)
                            # st.image(sky_segment, width=386)
                        
                            with st.form("업로드한 하늘 이미지로 합성 시작"):
                                state = check(segment_bytes, ref_mask_bytes)

                                if state == 'sky_image_dismatch':
                                    st.text("업로드한 하늘 이미지 합성 시 왜곡이 발생할 수 있습니다.")
                                elif state == 'input_image_no_sky':
                                    st.text("Dehazing 이미지에 하늘이 존재하지 않습니다.")
                                else:
                                    st.text("하늘을 합성할 수 있습니다.")

                                submitted = st.form_submit_button('하늘 합성 시작')
                        

                    else: # 선택한 구름에 대해 selection
                        selected = selection(dehaze_image_bytes, segment_bytes, option)   

                        if len(selected) == 0:
                            st.text("만족하는 하늘 이미지가 없습니다.")
                            st.text("직접 하늘 이미지를 업로드하거나 다른 키워드를 선택해주세요.")

                        if selected:
                            for i, select_sky_path in enumerate(selected):
                                st.image(select_sky_path, width=386, caption=str(i+1)+'번째', use_column_width='always')

                            with st.form("Select Cloud"):
                                select_option = st.selectbox(
                                '원하는 하늘을 선택해주세요',
                                [str(i+1)+'번째' for i in range(len(selected))])

                                selected_sky_path = selected[int(select_option[0])-1]
                                sky_image = Image.open(selected_sky_path)

                                mask_path = selected_sky_path.replace('img','mask')
                                ref_mask = Image.open(mask_path)
                                ref_mask_bytes = image_to_bytes(ref_mask)
                                
                                # 선택된 sky image bytes
                                sky_image_bytes = image_to_bytes(sky_image)

                                state = check(segment_bytes, ref_mask_bytes)

                                if state == 'sky_image_dismatch':
                                    st.text("선택한 하늘 이미지 합성 시 왜곡이 발생할 수 있습니다.")
                                elif state == 'input_image_no_sky':
                                    st.text("Dehazing 이미지에 하늘이 존재하지 않습니다.")
                                else:
                                    st.text("하늘을 합성할 수 있습니다.")

                                submitted = st.form_submit_button('하늘 합성 시작')
                            

                if submitted:
                    st.session_state.submitted = True
                    submitted = False

                ### 해당 버튼을 눌렀을 때, sky replacement가 시작된다.
                ### 추후에 각 이미지(3장) 별로 버튼을 만들면 될 듯.
                # sky_replace = None
                
                if 'submitted' in st.session_state and st.session_state.submitted:
                    print(st.session_state.submitted)
                    ##### SKY REPLACEMENT #####
                    sky_replace = replacement(dehaze_image_bytes, segment_bytes, sky_image_bytes, ref_mask_bytes)
                    # st.write(sky_replace)

                    # 공백
                    with col2:
                        st.write(' ')
                        
                    with col3:
                        st.header('After Cloud Generate')
                        # st.image(segment) # segmentation 확인하고 싶을 때
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

                    st.session_state.submitted = False # 초기화


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