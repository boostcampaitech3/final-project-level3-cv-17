from odmantic import Model


class ImageModel(Model):
    inputimage: str # 사용자가 업로드한 사진
    cloudimage: str # 어떤 구름 사진을 선택했는지

    class Config:
        collection = "users"