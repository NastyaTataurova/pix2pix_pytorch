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
    plt.subplot(1, 3, 1)
    plt.imshow(x[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=255)
    plt.title('Real')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=255)
    plt.title('Target')
    plt.axis('off')

    image_fake = np.asarray(y_gener[0].permute(1, 2, 0).cpu().detach().numpy(), dtype=np.float32)
    plt.subplot(1, 3, 3)
    plt.imshow(image_fake, vmin=0, vmax=255)
    plt.title('Fake')
    plt.axis('off')

  return history
