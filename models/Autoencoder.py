class IrtezaNet(nn.Module):
    def __init__(self):
        super(IrtezaNet,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, 
                      kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, 
                      kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU())
        self.decoder = nn.Sequential(             
            nn.Conv2d(64,32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32,3, kernel_size=1))
    
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 