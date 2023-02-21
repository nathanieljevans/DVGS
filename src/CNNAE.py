import torch 


class CNNAE(torch.nn.Module):
    '''
    CNN Autoencoder for images of size (in_channels,32,32)
    '''
    def __init__(self, in_channels=3, hidden_channels=16, latent_channels=4, act=torch.nn.ReLU, dropout=0.):
        super().__init__()
       
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            #torch.nn.BatchNorm2d(hidden_channels),
            #torch.nn.InstanceNorm2d(hidden_channels),            # [batch, 12, 16, 16]
            act(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            #torch.nn.BatchNorm2d(hidden_channels),
            #torch.nn.InstanceNorm2d(hidden_channels),           # [batch, 24, 8, 8]
            act(),
            torch.nn.Dropout(dropout),
			torch.nn.Conv2d(hidden_channels, latent_channels, 4, stride=2, padding=1),
            #torch.nn.BatchNorm2d(latent_channels),
            #torch.nn.InstanceNorm2d(latent_channels),           # [batch, 48, 4, 4]
            act(),
            torch.nn.Dropout(dropout),
        )
        self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(latent_channels, hidden_channels, 4, stride=2, padding=1),
            #torch.nn.BatchNorm2d(hidden_channels),
            #torch.nn.InstanceNorm2d(hidden_channels),  # [batch, 24, 8, 8]
            act(),
            torch.nn.Dropout(dropout),
			torch.nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            #torch.nn.BatchNorm2d(hidden_channels),
            #torch.nn.InstanceNorm2d(hidden_channels),  # [batch, 12, 16, 16]
            act(),
            torch.nn.Dropout(dropout),
            torch.nn.ConvTranspose2d(hidden_channels, in_channels, 4, stride=2, padding=1)   # [batch, 3, 32, 32]
        )

    def turn_batchnorm_off_(self): 
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def reset_parameters(self): 
        self.apply(weight_reset)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()