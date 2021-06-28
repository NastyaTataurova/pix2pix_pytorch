# pix2pix_pytorch
Реализована архитектура pix2pix, обученная наборах данных Maps и Flowers.

У меня не было опыта в разработке репозиторий github, поэтому код ниже может криво работать (или что-то не заработать), на этот случай добавляю ссылку на GoogleColab в pix2pix.ipynb и инструкцию к нему ниже.

Скачать проект:
```bash
git clone https://github.com/NastyaTataurova/pix2pix_pytorch
```
```bash
pip install -r ./pix2pix_pytorch/requirements.txt
```
## Maps
Примеры сгенерированных изображений:
![image](https://user-images.githubusercontent.com/49210968/123641026-75b5b200-d82a-11eb-8863-cd958276c591.png)

Загрузить обученную модель:
```bash
bash ./pix2pix_pytorch/bin/load_model_maps.sh
```
Запустить обученную модель для генерации изображения:
```bash
python3 src/generate_maps.py
```
Чтобы загрузить изображение, его нужно добавить в ./crs/images/, затем прописать путь.
![image](https://user-images.githubusercontent.com/49210968/123647427-bca6a600-d830-11eb-8a11-1e7d8802aa52.png)

## Flowers
Примеры сгенерированных изображений:

![image](https://user-images.githubusercontent.com/49210968/123640905-4e5ee500-d82a-11eb-9e71-11ca867c40bc.png)

![image](https://user-images.githubusercontent.com/49210968/123641070-8403ce00-d82a-11eb-8101-3b47f7a0259e.png)

Загрузить обученную модель:
```bash
bash ./pix2pix_pytorch/bin/load_model_flowers.sh
```
Запустить обученную модель для генерации изображения:
```bash
python3 src/generate_flowers.py
```
Чтобы загрузить изображение, его нужно добавить в ./crs/images/, затем прописать путь.
![image](https://user-images.githubusercontent.com/49210968/123648030-4787a080-d831-11eb-93ce-47789ac769f3.png)
