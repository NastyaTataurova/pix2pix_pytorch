from crs.model.generator import *
from crs.model.discriminator import *
from crs.utils.save_load_weights import *


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


    print('файлы есть вот тут /home/nastya/photo.png')
    dir = input("напши дир с файлом") #/home/nastya/photo.png
    #TODO
