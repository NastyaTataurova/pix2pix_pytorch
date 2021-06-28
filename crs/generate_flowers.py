from crs.model.generator import *
from crs.model.discriminator import *
from crs.utils.save_load_weights import *

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