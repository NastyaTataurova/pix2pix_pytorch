from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class GetDatasetMaps(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.files_names = os.listdir(file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        file_name = self.files_names[idx]
        file_path = os.path.join(self.file_paths, file_name)
        image = Image.open(file_path)
        image = np.array(image)

        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image
