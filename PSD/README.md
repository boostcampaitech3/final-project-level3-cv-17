# PSD 사용 시 참고사항

### 0. Environment
python==3.8.5  
torch==1.7.1  
torchvision==0.8.2  
cuda version : 11  

### 1. data 
data 폴더를 꼭 만들어주셔서 실험하고자하는 데이터를 넣어주셔야 합니다.
데이터 구조 예시는 아래와 같습니다.
```
├── data
│   ├── baek  
│   ├── crawling 
│   └── MRFID
│       ├── clear
│       ├── fog
``` 

제가 따로 추가한 코드는 
PSD/pretrained_model # ETCDataset 추가
PSD/make_data.py # 모든 데이터로 테스트해보기 시간이 오래걸려서 제가 각 폴더별로 7개씩인가? 데이터를 샘플링해서 새로운 데이터폴더를 만드는 코드입니다.(사용안하셔도 됩니다.)
