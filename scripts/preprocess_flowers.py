from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2


class GetDatasetFlowers(Dataset):
    def __init__(self, file_paths_jpg, file_paths_trimaps, transform):
        self.file_paths_jpg = file_paths_jpg
        self.file_paths_trimaps = file_paths_trimaps

        self.files_names_jpg = os.listdir(file_paths_jpg)
        self.files_names_trimaps = os.listdir(file_paths_trimaps)

        self.transform = transform

    def __len__(self):
        return len(self.file_paths_jpg)

    def __getitem__(self, idx):
        file_name_trimaps = self.files_names_trimaps[idx]
        file_name_jpg = file_name_trimaps[:11] + 'jpg'

        file_path_jpg = os.path.join(self.file_paths_jpg, file_name_jpg)
        image_target = Image.open(file_path_jpg)
        image_target = np.array(image_target)

        file_path_trimaps = os.path.join(self.file_paths_trimaps, file_name_trimaps)
        image_trimaps = cv2.imread(file_path_trimaps)
        image_trimaps = np.array(image_trimaps)

        image_target = self.transform(image_target)
        image_trimaps = self.transform(image_trimaps)

        return image_trimaps, image_target
