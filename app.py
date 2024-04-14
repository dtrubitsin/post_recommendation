import os
import pandas as pd

from fastapi import FastAPI
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from schema import PostGet

from catboost import CatBoostClassifier

if __name__ == '__main__':
    load_dotenv()

app = FastAPI()


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
        "/Users/dmitry/Documents/code/Start_ML/module_2/final_project/model/catboost_1_1")

    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path, format='cbm')
    return model


# Основная функция загрузки признаков из БД
def load_features():
    # Признаки по постам (созданные)
    logger.info("loading post feachures")
    post_feachures = pd.read_sql(
        """
        SELECT *
        FROM d_trubitsin_post_feachures_base 
        """,
        con=os.environ["POSTGRES_CONN"]
    )

    # Признаки по пользователям
    logger.info("loading user feachures")
    user_feachures = pd.read_sql(
        """
        SELECT *
        FROM public.user_data
        """,
        con=os.environ["POSTGRES_CONN"]
    )
    return [post_feachures, user_feachures]


logger.info("loading model")
model = load_models()

logger.info("loading feachures")
feachures = load_features()
logger.info("service is up and running")


# Эндпоинт для рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:

    # Признаки пользователя
    logger.info(f"user_id: {id}")
    logger.info("reading feachures")
    user_feachures = feachures[1].loc[feachures[1]["user_id"] == id].copy()
    user_feachures.drop('user_id', axis=1, inplace=True)

    # Признаки постов
    logger.info("dropping columns")
    post_feachures = feachures[0].drop(["index", "text"], axis=1)
    content = feachures[0][["post_id", "text", "topic"]]

    # Объединение признаков
    logger.info("zipping everything")
    add_user_feachures = dict(
        zip(user_feachures.columns, user_feachures.values[0]))
    request_data = post_feachures.assign(**add_user_feachures)
    request_data = request_data.set_index('post_id')

    # Добавление даты
    logger.info("adding time info")
    request_data['weekday'] = time.weekday()
    request_data['hour'] = time.hour

    # Предсказание вероятности для 1 класса (вероятнее всего понравится)
    logger.info("predicting")
    probabilities = model.predict_proba(request_data)[:, 1]
    request_data['prediction'] = probabilities

    # Получение топ-5 индексов вероятностей
    recommended_posts = request_data.sort_values("prediction")[-limit:].index

    return [PostGet(**{
        'id': i,
        'text': content[content['post_id'] == i]['text'].values[0],
        'topic': content[content['post_id'] == i]['topic'].values[0]
    }) for i in recommended_posts
    ]
