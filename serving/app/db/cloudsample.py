from odmantic import Model

### 구름 예시 사진용 db
class SampleModel(Model):
    cloudimage: str # 구름 예시 사진 경로
    cloudoption: str # 구름 예시 사진 옵션

    class Config:
        collection = "data"