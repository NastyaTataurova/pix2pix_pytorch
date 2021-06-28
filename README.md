# pix2pix_pytorch
Реализована архитектура pix2pix, обученная наборах данных Maps и Flowers.

У меня не было опыта в разработке репозиторий github, поэтому может возникнуть ситуация, что что-то пойдет не так (код будет криво работать или что-то не заработает), на этот случай добавляю ссылку на GoogleColab в pix2pix.ipynb и инструкцию к нему ниже.

# 1. Запуск с GitHub
Скачать проект:
```bash
git clone https://github.com/NastyaTataurova/pix2pix_pytorch
```
```bash
pip install -r ./requirements.txt
```
## 1.1 Maps
Примеры сгенерированных изображений:
![image](https://user-images.githubusercontent.com/49210968/123641026-75b5b200-d82a-11eb-8863-cd958276c591.png)

Загрузить обученную модель:
```bash
bash ./bin/load_model_maps.sh
```
Запустить обученную модель для генерации изображения:
```bash
python3 src/generate_maps.py
```
Чтобы загрузить изображение, его нужно добавить в ./crs/images/, затем прописать путь.
![image](https://user-images.githubusercontent.com/49210968/123647427-bca6a600-d830-11eb-8a11-1e7d8802aa52.png)

## 1.2 Flowers
Примеры сгенерированных изображений:

![image](https://user-images.githubusercontent.com/49210968/123640905-4e5ee500-d82a-11eb-9e71-11ca867c40bc.png)

![image](https://user-images.githubusercontent.com/49210968/123641070-8403ce00-d82a-11eb-8101-3b47f7a0259e.png)

Загрузить обученную модель:
```bash
bash ./bin/load_model_flowers.sh
```
Запустить обученную модель для генерации изображения:
```bash
python3 src/generate_flowers.py
```
Чтобы загрузить изображение, его нужно добавить в ./crs/images/, затем прописать путь.
![image](https://user-images.githubusercontent.com/49210968/123648030-4787a080-d831-11eb-93ce-47789ac769f3.png)

# 2. Запуск в GoogleColab
## 2.1 Загрузка обученной модели
### 2.1.1 Чтобы загрузить обученную модель и сгенерировать изображение для датасета Maps:
1. Запустить все до молуля "Dataset 1 -- Maps"
2. Запустить все в модуле "Train model 1 (Maps)"
3. Выбрать "y", когда предложат загрузить модель

![image](https://user-images.githubusercontent.com/49210968/123666301-c2f14e00-d841-11eb-8f1e-635189024c21.png)

4. Запустить модуль "Upload your image (Maps)", чтобы добавить свое изображение выбрать "1", дефолтное - "2"
5. Плучить результат

![image](https://user-images.githubusercontent.com/49210968/123666636-119ee800-d842-11eb-8142-f58644a6f5cc.png)
### 2.1.2 Аналогично с датасотом Flowes
Результат:

![image](https://user-images.githubusercontent.com/49210968/123667326-b8838400-d842-11eb-9ed3-f93d5e05cebc.png)

## 2.2 Запуск обучения модели
### 2.2.1 Чтобы загрузить обученную модель и сгенерировать изображение для датасета Maps:
1. Запустить все до молуля "Dataset 1 -- Maps"
2. Запустить все в модуле "Dataset 1 -- Maps"
3. Запустить все в модуле "Train model 1 (Maps)"
4. Выбрать "n", когда предложат загрузить модель
5. Начнется обучение
6. В модуле "Results (Maps)" можно посмотреть результаты на тестовых картинках
7. В модуле "Saving weights (Maps)" можно сохранить получившиеся веса
### 2.2.2 Аналогично с датасотом Flowes

