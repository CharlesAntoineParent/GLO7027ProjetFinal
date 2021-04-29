

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=1) # out: c x 7 x 7
        self.fc = nn.Linear(in_features=c*7*7, out_features=latent_dims)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*7*7)
        self.conv3 = nn.ConvTranspose2d(in_channels=c, out_channels=c*2, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered
        return x



class LNN(nn.Module):
    def __init__(self, num_of_linear_layers, type):
        self.num_of_linear_layers = num_of_linear_layers
        super(LNN, self).__init__()

        if type == "linear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=latent_dims, out_features=latent_dims) for _ in range(num_of_linear_layers)])
            self.forward = self.forward_linear

        if type == "weight_share":
            self.linear_layer = nn.Linear(in_features=latent_dims, out_features=latent_dims)
            self.forward = self.forward_weight_share

        if type == "fixed":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=latent_dims, out_features=latent_dims) for _ in range(num_of_linear_layers)])
            for layer in self.linear_layer:
                for param in layer.parameters():
                    param.requires_grad = False
            self.forward = self.forward_fixed

        if type == "nonlinear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=latent_dims, out_features=latent_dims) for _ in range(num_of_linear_layers)])
            self.forward = self.forward_nonlinear
        
    def forward_linear(self, x):
        for layer in self.linear_layer:
            x = layer(x)
        return x

    def forward_weight_share(self, x):
        for _ in range(self.num_of_linear_layers):
            x = self.linear_layer(x)
        return x
    
    def forward_fixed(self, x):
        for layer in self.linear_layer:
            x = layer(x)
        return x
    
    def forward_nonlinear(self, x):
        for layer in self.linear_layer:
            x = F.relu(layer(x))
        return x



class irma(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.lnn = LNN(*args, **kwargs)
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        min_rank_latent = self.lnn(latent)
        x_recon = self.decoder(min_rank_latent)
        return x_recon