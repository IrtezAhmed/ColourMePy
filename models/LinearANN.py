#baby ANN!

class IrtezaNet(nn.Module):
    def __init__(self):
        super(IrtezaNet,self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Linear(1,2),
            nn.Linear(2,3),
            nn.Linear(3,3))
                
    def forward(self,x):
        x = self.upsample(x)
        return x 