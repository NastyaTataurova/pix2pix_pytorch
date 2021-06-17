class discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv0 = nn.Sequential(
        nn.Conv2d(3 * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        nn.LeakyReLU(0.2)
    )
    
    self.conv1 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2)
    )
    
    self.conv2 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2)
    )
    
    self.conv3 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    
    self.conv4 = nn.Sequential(
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
    )
  
  def forward(self, x, y):
      x = torch.cat([x, y], dim=1)
      c0 = self.conv0(x)
      c1 = self.conv1(c0)
      c2 = self.conv2(c1)
      c3 = self.conv3(c2)
      c4 = self.conv4(c3)
      return c4
