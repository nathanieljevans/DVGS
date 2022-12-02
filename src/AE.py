'''autoencoder'''
import torch 
from NN import NN

class AE(torch.nn.Module):
    def __init__(self, in_channels, latent_channels, num_layers=1, norm=True, dropout=0., bias=True, act=torch.nn.ReLU): 
        ''''''
        super().__init__()

        self.encoder = NN(in_channels=in_channels, 
                            out_channels=latent_channels, 
                            num_layers=num_layers, 
                            hidden_channels=in_channels,
                            dropout=dropout,
                            norm=norm, 
                            bias=bias, 
                            act=act, 
                            out_fn=None)
        
        self.decoder = NN(in_channels=latent_channels, 
                            out_channels=in_channels, 
                            num_layers=num_layers, 
                            hidden_channels=in_channels,
                            dropout=dropout,
                            norm=norm, 
                            bias=bias, 
                            act=act, 
                            out_fn=None)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            for layer in self.children():
                weights_init(layer, gain=gain)

    def forward(self, x): 
        z = self.encoder(x)
        return self.decoder(z)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)