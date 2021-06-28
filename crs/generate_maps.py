from pix2pix_pytorch.crs.model.generator import *
from pix2pix_pytorch.crs.model.discriminator import *
from pix2pix_pytorch.crs.utils.save_load_weights import *
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as tt


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lr = 2e-4
    lambd = 100

    discrim_maps = discriminator().to(device)
    optimizer_discrim_maps = torch.optim.Adam(discrim_maps.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_bce_maps = nn.BCEWithLogitsLoss()

    gener_maps = generator().to(device)
    optimizer_gener_maps = torch.optim.Adam(gener_maps.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_l1_maps = nn.L1Loss()

    # LOAD WEIGHTS
    load_model_optimazer('discrim_maps.pth', discrim_maps, optimizer_discrim_maps, lr, device)
    load_model_optimazer('gener_maps.pth', gener_maps, optimizer_gener_maps, lr, device)


    print('Загрузка файла. Файл есть вот тут: <./images/map.jfif>')
    file_path = input('Напишите путь файлу: ')

    image_size = 256
    # stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transforms = tt.Compose([tt.ToPILImage(),
                             tt.Resize((256, 256)),
                             tt.ToTensor(),
                             # tt.Normalize(*stats)
                             ])

    # image generation
    image = Image.open(file_path)
    image = transforms(np.array(image))

    gener_image = gener_maps(image.unsqueeze(0).to(device))[0]
    gener_image = np.asarray(gener_image.permute(1, 2, 0).cpu().detach().numpy(), dtype=np.float32)

    # result
    im = Image.fromarray((gener_image * 255).astype(np.uint8))
    im.show()
