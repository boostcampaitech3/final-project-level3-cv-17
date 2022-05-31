from fastapi import FastAPI, UploadFile, File, Response
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.dehazing import get_prediction
from app.sky_replace import segmentor, replace_sky
import io

app = FastAPI()

@app.get("/")
def hello():
    return "Backend for Model and DB"

@app.post("/dehazing", description="hazing 결과를 요청합니다.")
async def dehazing_prediction(files: List[UploadFile] = File(...)):
    for file in files:
        image_bytes = await file.read()
        inference_result = get_prediction(image_bytes)

    img_byte_arr = io.BytesIO()
    inference_result.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/segmentor", description="sky segmentation 결과를 요청합니다.")
async def segmentation(files: List[UploadFile] = File(...)):
    for file in files:
        image_bytes = await file.read()
        segment_result = segmentor(image_bytes, True)

    img_byte_arr = io.BytesIO()
    segment_result.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/replace", description="sky replacement 결과를 요청합니다.")
async def replacement(files: List[UploadFile] = File(...)):
    dehaze_image_bytes = await files[0].read()
    mask_image_bytes = await files[1].read()
    sky_image_bytes = await files[2].read()

    final = replace_sky(dehaze_image_bytes, mask_image_bytes, sky_image_bytes)

    img_byte_arr = io.BytesIO()
    final.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")
