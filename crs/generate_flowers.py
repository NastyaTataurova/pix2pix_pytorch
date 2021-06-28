from pix2pix_pytorch.crs.model.generator import *
from pix2pix_pytorch.crs.model.discriminator import *
from pix2pix_pytorch.crs.utils.save_load_weights import *
from PIL import Image
import numpy as np
import torchvision.transforms as tt
import cv2


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lr = 2e-4
    lambd = 100

    discrim_flowers = discriminator().to(device)
    optimizer_discrim_flowers = torch.optim.Adam(discrim_flowers.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_bce_flowers = nn.BCEWithLogitsLoss()

    gener_flowers = generator().to(device)
    optimizer_gener_flowers = torch.optim.Adam(gener_flowers.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_l1_flowers = nn.L1Loss()

    load_model_optimazer('discrim_flowers.pth', discrim_flowers, optimizer_discrim_flowers, lr, device)
    load_model_optimazer('gener_flowers.pth', gener_flowers, optimizer_gener_flowers, lr, device)

    print('Загрузка файла. Файл есть вот тут: <./images/flower.png>')
    file_path = input('Напишите путь файлу: ')  # input("напши дир с файлом ") #
    import os

    print(os.getcwd())
    image_size = 256
    # stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transforms = tt.Compose([tt.ToPILImage(),
                             tt.Resize((256, 256)),
                             tt.ToTensor(),
                             # tt.Normalize(*stats)
                             ])

    # image generation
    image = cv2.imread(file_path)
    image = transforms(np.array(image))

    gener_image = gener_flowers(image.unsqueeze(0).to(device))[0]
    gener_image = np.asarray(gener_image.permute(1, 2, 0).cpu().detach().numpy(), dtype=np.float32)

    # result
    im = Image.fromarray((gener_image * 255).astype(np.uint8))
    im.show()
