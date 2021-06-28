from torch.utils.data import DataLoader
from pix2pix_pytorch.scripts.preprocess_maps import *
import torchvision.transforms as tt

image_size = 256
# stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transforms = tt.Compose([tt.ToPILImage(),
                         tt.Resize((256, 256)),
                         tt.ToTensor(),
                         # tt.Normalize(*stats)
                         ])

# path to images
train_path_maps = './datasets/maps/train'
val_path_maps = './datasets/maps/val'

# creating a train dataloader
train_dataset_maps = GetDatasetMaps(file_paths=train_path_maps, transform=transforms)
train_loader_maps = DataLoader(train_dataset_maps, batch_size=1)

# creating a val dataloader
val_dataset_maps = GetDatasetMaps(file_paths=val_path_maps, transform=transforms)
val_loader_maps = DataLoader(val_dataset_maps, batch_size=5)
