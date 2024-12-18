Что было сделано?
1. Провела базоdый EDA, построила необходимые визуализации, чтобы лучше понять данные и их связь с целевой переменной selling_price.
2. Провела препроцессинг данных, в частности удалила дубликаты, заполнила пропуски медианными значениями, сгенерировала новые признаки (например, из признака name получилось 3 признака: brand, model, series).
3. Построила несколько моделей линейной регрессии: классическая линейная модель на сырых данных (R2 for test: 0.59) и на стандартизированных данных (R2 for test: 0.594), Lasso-регрессия (в том числе с использованием GridSearchCV; R2 for test: 0.594), ElasticNet (R2 for test: 0.5733), Ridge-регрессия с добавлением категориальных фичей (R2 for test: 0.9166). На этапе построения последней моделе я обогатила датафрейм обработанными категориальными признаками, что значительно улучшило и MSE, и R2 (так, модель описывала 92% данных на тесте).
4. Реализовала бизнес-метрику в соответствии с заданием. Метрика показывает для данной модели МО долю прогнозов, отличающихся от реальных цен на эти авто не более чем на 10%. По ней лучшей моделью оказалась линейная регрессия на сырых данных.
5. Реализовала сервис на FastApi для двух случаев: когда на вход подаются признаки одного объекта и когда на вход подается список признаков для нескольких объектов.

Из того, что хотелось бы доработать:
- лучше поработать над признаками, например, лучше обработать категориальный признак name;
- оптимизировать код для работы сервиса. Поскольку писала сервис впервые, скорее всего некоторые моменты в коде неоптимальны, хотелось бы плотнее с этим поработать.
