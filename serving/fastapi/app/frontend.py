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

### front-end 화면 구성 바꾸기
### 2대 2 화면 구성
### 상단은 input 과 dehazed 결과 (2개)
### 하단은 구름 예시 사진들과 구름 합성 최종 사진 (2개)

@st.cache(ttl=50)
def dehazing(uploaded_file, image_bytes): # user upload image
    files = [
        ('files', (uploaded_file.name, image_bytes,
                    uploaded_file.type))
    ]
    response = requests.post("http://localhost:30001/dehazing", files=files)
    dehaze_image_bytes = response.content 
    dehaze_image = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    return dehaze_image_bytes, dehaze_image # dehazed image

@st.cache(ttl=50)
def segmentation(image_bytes): # dehazed image
    files = [
        ('files', (image_bytes))
    ]
    response = requests.post("http://localhost:30001/segmentor", files=files)
    segment_bytes = response.content 
    segment_image = Image.open(io.BytesIO(response.content)).convert('L') # 'L' Grayscale
    
    return segment_bytes, segment_image # segment sky image

@st.cache(ttl=50)
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
    image.save(buf, format="JPEG") # JPG, PNG, JPEG
    bytes_img = buf.getvalue()
    return bytes_img

@st.cache(ttl=50)
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

    st.write(f"{cloud_option}으로 다운로드")
    db_save(cloud_option, files)
    

def main():
    st.title("Dehazing Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        ### 상단 부분
        with st.container():
            col1, col2, col3 = st.columns([3,1,3])

            with col1:
                st.header("Before Dahazing")
                st.image(image, caption='Uploaded Image')

                ### 현재 사용자가 사진을 업로드한 상태
                ### 이 때, background에서 dehazing inference를 진행하면 된다.
                ##### DEHAZING MODEL #####
                dehaze_image_bytes, dehaze_image = dehazing(uploaded_file, image_bytes)
            
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
                        ' ',
                        ('작은 구름', '큰 구름', '분홍 구름'))

                    ### db에서 이미지 데이터를 가져온다. (SampleModel db)
                    ### 버튼을 누르면 requests post를 날리고 -> main.py (app.post("/cloud")로 보내면 될 듯?)
                    ### 이미지를 response 받을 때, image들이 들어가 있는 리스트를 받을 듯
                    ### response로 받은 image list를 여기서 for문으로 풀어서 st.image하면 될 듯
                    # cloud_response = requests.post("http://localhost:30001/cloud")
                    # st.write(cloud_response.content[0])
                    # st.image(cloud_response.content[0], caption='for문으로 이미지 출력')
                    
                    # for i in cloud_response:
                    #     st.image(i)
                    # option 바꿀 때마다 다시 실행
                    if option == '작은 구름':
                        st.image("/opt/ml/input/data/Sky/sky001.jpg", caption = '작은 구름', width=384)
                        sky_image = Image.open("/opt/ml/input/data/Sky/sky001.jpg")
                    elif option == '큰 구름':
                        st.image("/opt/ml/input/data/Sky/sky002.jpg", caption = '큰 구름', width=384)
                        sky_image = Image.open("/opt/ml/input/data/Sky/sky002.jpg")
                    elif option == '분홍 구름':
                        st.image("/opt/ml/input/data/Sky/sky003.jpeg", caption = '분홍 구름', width=384)
                        sky_image = Image.open("/opt/ml/input/data/Sky/sky003.jpeg")

                    

                    # 선택된 sky image bytes
                    sky_byte_arr = image_to_bytes(sky_image)

                # 이미지 버튼 시도
                # st.markdown(
                #     "![this is an image link](https://img.freepik.com/free-photo/white-cloud-on-blue-sky-and-sea_74190-4488.jpg?w=2000){:width=100 height=100}"
                # )

                # img = Image.open("images/top_spiderman.png")
                # st.button(st.image(img))

                ### 해당 버튼을 눌렀을 때, sky replacement가 시작된다.
                ### 추후에 각 이미지(3장) 별로 버튼을 만들면 될 듯.
                # sky_replace = None
                if st.button('구름 합성 시작'):
                    ##### SKY REPLACEMENT #####
                    sky_replace = replacement(dehaze_image_bytes, segment_bytes, sky_byte_arr)
                    # st.write(sky_replace)

                    # 공백
                    with col2:
                        st.write(' ')

                    
                    with col3:
                        st.header('After Cloud Generate')

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
            st.header('예시 사진')
            col0, col1, col2, col3 = st.columns([1,2,2,1])
            with col1:
                st.header("Before Dehazing")
                st.image("/opt/ml/input/data/Hidden/hazy/001_baek.jpg")
            with col2:
                st.header("After Dehazing")
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
