import os
import pandas as pd

from fastapi import FastAPI

from typing import List
from datetime import datetime
from loguru import logger
import hashlib

from schema import PostGet, Response

from catboost import CatBoostClassifier

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# Константы
salt = 'get_exp_group'
num_groups = 2


# Разбиение пользователей на группы
def get_exp_group(user_id: int) -> str:
    exp_group = int(hashlib.md5((str(user_id) + salt).encode()
                                ).hexdigest(), 16) % num_groups

    if exp_group == 0:
        return 'control'
    else:
        return 'test'


# Путь модели
def get_model_path(model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if model_name == 'model_control':
            MODEL_PATH = '/workdir/user_input/model_control'
        elif model_name == 'model_test':
            MODEL_PATH = '/workdir/user_input/model_test'
        else:
            raise ValueError('unknown name')

    else:
        if model_name == 'model_control':
            MODEL_PATH = '/Users/dmitry/Documents/code/Start_ML/module_2/final_project/model/catboost_ml_1'
        elif model_name == 'model_test':
            MODEL_PATH = '/Users/dmitry/Documents/code/Start_ML/module_2/final_project/model/catboost_dl_3'
        else:
            raise ValueError('unknown name')
    return MODEL_PATH


# Загрузка модели
def load_models(model_name: str):
    model_path = get_model_path(model_name)

    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path, format='cbm')
    return model


# Основная функция загрузки признаков из БД
def load_features():
    # Признаки по постам (созданные) TF-IDF
    logger.info("loading tf-idf post feachures")
    post_feachures_control = pd.read_sql(
        """
        SELECT *
        FROM d_trubitsin_post_feachures_base 
        """,
        con=os.environ["POSTGRES_CONN"]
    )

    # Признаки по постам (созданные) DL
    logger.info("loading dl post feachures")
    post_feachures_test = pd.read_sql(
        """
        SELECT *
        FROM d_trubitsin_post_feachures_dl 
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
    return [post_feachures_control, post_feachures_test, user_feachures]


# Выдача рекомендаций
def get_recommendations(id, time, limit, model, post_feachures, user_feachures) -> List[PostGet]:

    # Признаки пользователя
    logger.info(f"user_id: {id}")
    logger.info("reading feachures")
    user_feach = user_feachures.loc[user_feachures["user_id"] == id].copy()
    user_feach.drop(['user_id', 'os', 'source'], axis=1, inplace=True)

    # Признаки постов
    logger.info("dropping columns")
    post_feach = post_feachures.drop(["index", "text"], axis=1)
    content = post_feachures[["post_id", "text", "topic"]]

    # Объединение признаков
    logger.info("zipping everything")
    add_user_feachures = dict(
        zip(user_feach.columns, user_feach.values[0]))
    request_data = post_feach.assign(**add_user_feachures)
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
    }) for i in recommended_posts]


logger.info("loading model")
model_control = load_models("model_control")
model_test = load_models("model_test")

logger.info("loading feachures")
feachures = load_features()
logger.info("service is up and running")


# Эндпоинт для рекомендаций
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> Response:
    # Отнесение пользователя к группе
    logger.info('getting exp group')
    exp_group = get_exp_group(id)
    logger.info(f"User '{id}' assigned to '{exp_group}'")

    if exp_group == 'control':
        recommendations = get_recommendations(
            id, time, limit, model_control, feachures[0], feachures[2])
    elif exp_group == 'test':
        recommendations = get_recommendations(
            id, time, limit, model_test, feachures[1], feachures[2])
    else:
        raise ValueError('unknown group')

    return Response(exp_group=exp_group, recommendations=recommendations)
