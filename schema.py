from pydantic import BaseModel
import datetime


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
