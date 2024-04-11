# Система Рекомендации Постов
Этот репозиторий содержит систему рекомендаций постов, которая предлагает пользователю соответствующие посты на основе их предпочтений и поведения. Система рекомендаций разработана для увеличения вовлеченности пользователей и удовлетворенности, предоставляя персонализированные рекомендации контента. В качестве входных данных были данные о пользователе, постах и взаимодействия пользователей с постами.

Проект создан в рамках курса StartML на платформе Karpov.Courses

## Конфигурация
- База Данных: PostgreSQL
- Алгоритм Рекомендаций: Контентный
- Модель: Catboost
- Фреймворк: FastAPI

## Описание файлов
- FP_endpoint – ноутбук для тестирования эндпоинта представленного в файле app.py
- FP_feachures – ноутбук для работы с данными, генерации признаков и обучения модели
- FP_load_dp – ноутбук для загрузки признаков в базу данных
- FP_vector – ноутбук для векторизации текста с использованием нейронной сети jina-embeddings-v2-small-en
- app.py – файл с эндпоинтом