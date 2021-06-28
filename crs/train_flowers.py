from tqdm import tqdm
from crs.model.generator import *
from crs.model.discriminator import *
from crs.utils.save_load_weights import *
from crs.data.dataset_flowers import *


def train(discrim, gener, loader, opt_discrim, opt_gener, loss_l1, loss_bce, num_epoch):
  history = []
  for epoch in tqdm(range(num_epoch)):    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        y_gener = gener(x)
        discrim_real_image = discrim(x, y)
        discrim_real_image_loss = loss_bce(discrim_real_image, torch.ones_like(discrim_real_image))
        discrim_gener_image = discrim(x, y_gener.detach())
        discrim_gener_image_loss = loss_bce(discrim_gener_image, torch.zeros_like(discrim_gener_image))
        discrim_loss = (discrim_real_image_loss + discrim_gener_image_loss) / 2

        opt_discrim.zero_grad()
        discrim_loss.backward()
        opt_discrim.step()

        # Train Generator
        discrim_gener_image = discrim(x, y_gener)
        G_fake_loss = loss_bce(discrim_gener_image, torch.ones_like(discrim_gener_image))
        L1 = loss_l1(y_gener, y) * lambd
        G_loss = G_fake_loss + L1

        opt_gener.zero_grad()
        G_loss.backward()
        opt_gener.step()

        # clear_output(wait=True)

    history.append((discrim_real_image_loss, discrim_gener_image_loss, discrim_loss))

    print(f'\n{epoch+1}/{num_epoch} epoch: loss={discrim_loss}, generated image loss={discrim_gener_image_loss}')
    return history

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

    num_epoch = 400
    train(discrim_flowers, gener_flowers, train_loader_flowers, optimizer_discrim_flowers, optimizer_gener_flowers,
          loss_l1_flowers, loss_bce_flowers, num_epoch)

    # saving weights
    save_model_optimazer(discrim_flowers, optimizer_discrim_flowers, 'discrim_flowers.pth')
    save_model_optimazer(gener_flowers, optimizer_gener_flowers, 'gener_flowers.pth')
