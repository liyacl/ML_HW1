# ML Homework 1

В данном домашнем задании реализовано следующее:
- базовый разведочный анализ (EDA);
- препроцессинг данных;
- обучение различных вариаций линейной регресии (классическая линейная регрессия, Lasso- и Ridge-регрессия);
- сравнение качества всех моделей (MSE и R2), в результате которого выяснилось, что лучшей является Ridge-регрессия, обученная на стандартизированных признаках + обработанных категориальных фичах;
- сервис на FastApi для двух случаев: когда на вход в формате json подаются признаки одного объекта и когда на вход подается csv-файл с признаками тестовых объектов.

### Скриншоты с работоспособностью сервиса
см. https://docs.google.com/document/d/1Dn1QZUzQ9rpULzFsumOq1NHZFqkzv-zuRjt_HULWNP4/edit?usp=sharing
