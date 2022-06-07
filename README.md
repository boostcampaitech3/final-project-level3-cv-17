## 미세먼지 없는 맑은 사진
- Naver Boostcamp AI Tech 3rd final project
- CV 17조 MG세대
- Demo Video :
- Presentation Slide : 

## Project Abstract
* Problem Definition
    
    * 특별한 날, 특별한 장소에서 미세먼지 때문에 원하는 사진을 찍지 못하거나 안 찍는 상황이 생김
    * 하지만 보정에 대한 전문 지식이 부족하거나 필터가 제한되는 경우 사용자가 원하는 방향으로 사진을 보정하기 어려웠음

* Main features

    * 사용자가 업로드한 Hazy한 이미지를 미세먼지가 없는 선명한 사진으로 변환
    * 사용자는 원하는 Keyword의 하늘 이미지를 업로드한 사진에 합성

## Member Introduction
|팀원|Github|역할|
| :--------: | :--------: | :--------: |
|[T3078] 민선아||Image Dehazing|
|[T3101] 백경륜||Product Serving|
|[T3139] 이도연||Product Serving|
|[T3177] 이효석||PM, Sky Replacement|
|[T3179] 임동우|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|Image Dehazing|

## Service Architecture
![image](https://user-images.githubusercontent.com/81875412/172397327-77f34979-b0b4-45f7-992f-b0e126c6d10b.png)

## Streamlit & Fastapi Demo
- Run

## Model Process
![image](https://user-images.githubusercontent.com/81875412/172397492-34a7450e-32e4-4f45-a9a2-87b4a43a07f2.png)

## Image Dehazing
- Pretrain
- Finetune

## Sky Replacement
- Train
