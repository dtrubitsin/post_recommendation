# Система Рекомендации Постов
Этот репозиторий содержит систему рекомендаций постов, которая предлагает пользователю соответствующие посты на основе их предпочтений и поведения. Система рекомендаций разработана для увеличения вовлеченности пользователей и удовлетворенности, предоставляя персонализированные рекомендации контента. В качестве входных данных были данные о пользователе, постах и взаимодействия пользователей с постами. 

Реализованы 2 модели:
- model_control = Модель, обученная на признаках, полученных из текста с помощью tf-idf
- model_test = Модель, обученная на признаках, полученных из текста с помощью трансформера jina-embeddings

Реализовано проведение A/B эксперимента.

Проект создан в рамках курса StartML на платформе Karpov.Courses

## Конфигурация
- База Данных: PostgreSQL
- Алгоритм Рекомендаций: Контентный
- Модель: Catboost
- Фреймворк: FastAPI

## Описание файлов
- FP_endpoint – ноутбук для тестирования эндпоинта представленного в файле app.py
- FP_feachures – ноутбук для работы с данными, генерации признаков и обучения модели
- FP_vector – ноутбук для векторизации текста с использованием нейронной сети трансформера jina-embeddings-v2-small-en
- app.py – файл с эндпоинтом