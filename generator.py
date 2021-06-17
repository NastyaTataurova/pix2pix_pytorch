class generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv0 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
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
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)        
    )

    self.conv456 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)        
    )

    self.bottleneck = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
        nn.ReLU(0.2)        
    )

    self.dec_conv0 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(0.2),
        nn.Dropout(0.5)        
    )

    self.dec_conv12 = nn.Sequential(
        nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(0.2),
        nn.Dropout(0.5)        
    )

    self.dec_conv3 = nn.Sequential(
        nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(0.2)       
    )

    self.dec_conv4 = nn.Sequential(
        nn.ConvTranspose2d(512 * 2, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(0.2)       
    )

    self.dec_conv5 = nn.Sequential(
        nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(0.2)       
    )

    self.dec_conv6 = nn.Sequential(
        nn.ConvTranspose2d(128 * 2, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(0.2)       
    )

    self.dec_conv7 = nn.Sequential(
        nn.ConvTranspose2d(64 * 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Sigmoid() # nn.Tanh()       
    )

  def forward(self, x):
    # encoder
    e0 = self.conv0(x) # 128  
    e1 = self.conv1(e0) # 64  
    e2 = self.conv2(e1) # 32  
    e3 = self.conv3(e2) # 16  
    e4 = self.conv456(e3) # 8  
    e5 = self.conv456(e4) # 4  
    e6 = self.conv456(e5) # 2  

    # bottleneck
    b = self.bottleneck(e6) # 1

    # decoder
    d0 = self.dec_conv0(b) # 2  
    d1 = self.dec_conv12(torch.cat((d0, e6), dim=1)) # 4
    d2 = self.dec_conv12(torch.cat((d1, e5), dim=1)) # 8  
    d3 = self.dec_conv3(torch.cat((d2, e4), dim=1)) # 16  
    d4 = self.dec_conv4(torch.cat((d3, e3), dim=1)) # 32  
    d5 = self.dec_conv5(torch.cat((d4, e2), dim=1)) # 64  
    d6 = self.dec_conv6(torch.cat((d5, e1), dim=1)) # 128  
    d7 = self.dec_conv7(torch.cat((d6, e0), dim=1)) # 256  
    return d7
