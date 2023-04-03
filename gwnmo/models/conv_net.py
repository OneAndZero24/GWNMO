class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True, pool=False):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if pool:
            trunk.append(nn.AdaptiveAvgPool2d((1,1)))

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim: int = outdim

    def forward(self,x):
        out = self.trunk(x)
        return out