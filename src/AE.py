'''autoencoder'''
import torch 
from NN import NN

class AE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_layers=1, norm=True, dropout=0., bias=True, act=torch.nn.ReLU): 
        ''''''
        super().__init__()

        self.encoder = NN(in_channels=in_channels, 
                            out_channels=latent_channels, 
                            num_layers=num_layers, 
                            hidden_channels=hidden_channels,
                            dropout=dropout,
                            norm=norm, 
                            bias=bias, 
                            act=act, 
                            out_fn=None)
        
        self.decoder = NN(in_channels=latent_channels, 
                            out_channels=in_channels, 
                            num_layers=num_layers, 
                            hidden_channels=hidden_channels,
                            dropout=dropout,
                            norm=norm, 
                            bias=bias, 
                            act=act, 
                            out_fn=None)

    def reset_parameters(self): 
        self.apply(weight_reset)

    def forward(self, x): 
        z = self.encoder(x)
        x = self.decoder(z)
        return x

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()