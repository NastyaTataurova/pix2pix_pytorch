from torch.utils.data import DataLoader
from pix2pix_pytorch.scripts.preprocess_flowers import *
import torchvision.transforms as tt


image_size = 256
# stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transforms = tt.Compose([tt.ToPILImage(),
                         tt.Resize((256, 256)),
                         tt.ToTensor(),
                         # tt.Normalize(*stats)
                         ])

# path to train images
train_path_flowers_jpg = './datasets/flowers/train/jpg/jpg'
train_path_flowers_trimaps = './datasets/flowers/train/trimaps/trimaps'

# path to val images
val_path_flowers_jpg = './datasets/flowers/test/jpg/jpg'
val_path_flowers_trimaps = './datasets/flowers/test/trimaps/trimaps'

# creating a train dataloader
train_dataset_flowers = GetDatasetFlowers(file_paths_jpg=train_path_flowers_jpg,
                                          file_paths_trimaps=train_path_flowers_trimaps,
                                          transform=transforms)
train_loader_flowers = DataLoader(train_dataset_flowers, batch_size=1)

# creating a val dataloader
val_dataset_flowers = GetDatasetFlowers(file_paths_jpg=val_path_flowers_jpg,
                                        file_paths_trimaps=val_path_flowers_trimaps,
                                        transform=transforms)
val_loader_flowers = DataLoader(val_dataset_flowers, batch_size=5)
