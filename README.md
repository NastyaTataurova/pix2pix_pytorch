# pix2pix_pytorch
Implemented the pix2pix architecture, trained on the map dataset and flower dataset.

## Maps
![image](https://user-images.githubusercontent.com/49210968/123641026-75b5b200-d82a-11eb-8863-cd958276c591.png)

```bash
git clone https://github.com/NastyaTataurova/pix2pix_pytorch
```
```bash
pip install -r ./pix2pix_pytorch/requirements.txt
```
```bash
bash ./pix2pix_pytorch/bin/load_model_maps.sh
```
```bash
python3 src/generate_maps.py
```

## Flowers
![image](https://user-images.githubusercontent.com/49210968/123640905-4e5ee500-d82a-11eb-9e71-11ca867c40bc.png)
![image](https://user-images.githubusercontent.com/49210968/123641070-8403ce00-d82a-11eb-8101-3b47f7a0259e.png)

```bash
git clone https://github.com/NastyaTataurova/pix2pix_pytorch
```
```bash
pip install -r ./pix2pix_pytorch/requirements.txt
```
```bash
bash ./pix2pix_pytorch/bin/load_model_flowers.sh
```
```bash
python3 src/generate_flowers.py
```
