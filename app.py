import os
import pandas as pd
import numpy as np

from fastapi import FastAPI
from sqlalchemy import create_engine, Integer, Column, Text
from sqlalchemy.ext.declarative import declarative_base
from typing import List
from datetime import datetime

from dotenv import load_dotenv

from schema import PostGet

from catboost import CatBoostClassifier

if __name__ == '__main__':
    load_dotenv()

app = FastAPI()

Base = declarative_base()


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    topic = Column(Text)


# Путь модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

# Загрузка модели


def load_models():
    model_path = get_model_path(
        "/Users/dmitry/Documents/code/Start_ML/module_2/final_project/model/catboost_5_1")

    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path, format='cbm')
    return model


# Загрузка данных по чанкам
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(os.environ["POSTGRES_CONN"])
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Основная функция загрузки признаков из БД
def load_features(table: str = 'd_trubitsin_user_feachures_lesson_22') -> pd.DataFrame:
    data = batch_load_sql(f'SELECT * FROM {table}')
    return data


model = load_models()
# признаки пользователя
user_data = load_features('d_trubitsin_user_feachures_lesson_22')
post_data = load_features(
    'd_trubitsin_post_feachures_lesson_22')  # признаки постов
# таблица со всеми постами и исходной информацией
all_posts = load_features('public.post_text_df')

# эндпоинт для рекомендаций


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:

    # создание таблицы с данными для предсказаний
    request_data = pd.merge(
        user_data[user_data['user_id'] == id], post_data, how='cross')

    request_data['weekday'] = time.weekday()
    request_data['hour'] = time.hour

    request_data.drop(['user_id', 'post_id'], axis=1, inplace=True)

    # предсказание вероятности для 1 класса (вероятнее всего понравится)
    probabilities = model.predict_proba(request_data)[:, 1]

    # выбор 5 индексов наиболее вероятных постов
    top_5_indices = np.argsort(probabilities)[::-1][:limit]
    response = all_posts.iloc[top_5_indices].copy()

    # форматирование выдачи согласно шаблону
    response.rename(columns={'post_id': 'id'}, inplace=True)
    result_list = response.to_dict(orient='records')

    return result_list
