from fastapi import FastAPI, UploadFile, File, Response
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.dehazing import get_prediction
from app.sky_replace import segmentor, select_sky_paths, replace_sky
import io
import os

from PIL import Image

### db 관련 추가 import
from app.db import mongodb
from app.db.image import ImageModel # 사용자 이미지와 구름 정보 저장용
from app.db.cloudsample import SampleModel # 구름 예시 사진들 저장용

app = FastAPI()

@app.get("/")
def hello():
    return "Backend for Model and DB"

def image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG") # JPG, PNG, JPEG
    bytes_img = buf.getvalue()
    return bytes_img

@app.post("/dehazing", description="hazing 결과를 요청합니다.")
async def dehazing_prediction(files: List[UploadFile] = File(...)):
    image_bytes = await files[0].read() # user upload image
    # Dehazing get_prediction
    inference_result = get_prediction(image_bytes)
    img_byte_arr = image_to_bytes(inference_result)
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/segmentor", description="sky segmentation 결과를 요청합니다.")
async def segmentation(files: List[UploadFile] = File(...)):
    image_bytes = await files[0].read() # dehazed image
    segment_result = segmentor(image_bytes)
    img_byte_arr = image_to_bytes(segment_result)
    return Response(content=img_byte_arr, media_type="image/png")

class FilePath(BaseModel):
    file_path: List[str] = []

@app.post("/select/{cloud_option}", description="이미지와 어울리는 sky를 선택합니다.", response_model=FilePath)
async def sky_select(files: List[UploadFile] = File(...), cloud_option: str=None):
    dehaze_image_bytes = await files[0].read() # dehazed image
    mask_image_bytes = await files[1].read()
    selected = select_sky_paths(dehaze_image_bytes, mask_image_bytes, cloud_option)
    selected = FilePath(file_path=selected)
    return selected # selected image paths

@app.post("/replace", description="sky replacement 결과를 요청합니다.")
async def replacement(files: List[UploadFile] = File(...)):
    dehaze_image_bytes = await files[0].read() # dehazed image
    mask_image_bytes = await files[1].read()
    sky_image_bytes = await files[2].read() # sky image
    sky_mask_bytes = await files[3].read() # sky mask
    # Sky Replacement
    final = replace_sky(dehaze_image_bytes, mask_image_bytes, sky_image_bytes, sky_mask_bytes)
    img_byte_arr = image_to_bytes(final)
    return Response(content=img_byte_arr, media_type="image/png") # Final image(sky replaced image)


### 구름 카테고리 선택 (ex. 분홍 구름) -> db에서 3장의 이미지 가져옴 -> 웹에 출력
### 해당 부분은 제가 할 예정입니다.
@app.post("/cloud", description="cloud 이미지들을 요청합니다.")
async def cloud_order():    # 딱히 input은 없다
    # db에 있는 모든 사진 데이터 가져옴
    cloud_images = await mongodb.engine.find(SampleModel)
    print('db에 있는 데이터 :', cloud_images)
    # for문으로 하나씩 빼면 될 듯
    cloud_list = []
    for i in cloud_images:
        # print(i.cloudimage)
        cloud_list.append(i.cloudimage)
    # print('이미지 데이터만 가져오기', cloud_images[0].cloudimage)

    print(cloud_list)
    print(type(cloud_list[0]))
    return Response(content=cloud_list)

### 사용자 업로드 이미지와, 선택한 구름 정보 2가지를 DB에 저장
### 업로드 이미지 자체를 vscode 서버에 저장하고, DB에는 그에 대한 절대경로를 저장할 예정
@app.post("/save/{cloud_option}", description="cloud 이미지들을 요청합니다.")
async def save_order(files: List[UploadFile] = File(...), cloud_option: str=None):      # 사진 1장, string 값 1개 (총 2개를 받는다)
    ### 사진이랑 string을 db에 저장
    print('--------------post save 실행--------------------')
    dehaze_image_bytes = await files[0].read()
    print('구름 옵션:', cloud_option)

    # dehaze_image_bytes 이미지 자체를 저장
    haze_img = Image.open(io.BytesIO(dehaze_image_bytes))
    haze_img = haze_img.convert("RGB")

    base_url = '/opt/ml/input/final-project-level3-cv-17/data/input_image/'
    path, dirs, files = next(os.walk(base_url))
    file_count = len(files)

    img_url = base_url + 'input' + str(file_count+1) + '.png'
    haze_img.save(img_url, 'png')

    save_image = ImageModel(inputimage=img_url, cloudoption=cloud_option)
    await mongodb.engine.save(save_image) # db에 저장  
    print('-----save 완료-----')

    return {"cloud" : "images"}


@app.on_event("startup")
def on_app_start():
    """before app starts"""
    mongodb.connect()


@app.on_event("shutdown")
def on_app_shutdown():
    """after app shutdown"""
    mongodb.close()