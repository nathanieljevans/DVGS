import torch 
from NN import NN

class CNN(torch.nn.Module):
    def __init__(self, in_conv, out_conv, out_channels, kernel_size, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU, out_fn=None): 
        ''''''
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(in_conv, out_conv, kernel_size), 
                                        act(),
                                        torch.nn.MaxPool2d(2,2),
                                        torch.nn.Conv2d(out_conv, 3*out_conv, kernel_size), 
                                        act(),
                                        torch.nn.MaxPool2d(2,2))

        self.fc = NN(in_channels=None, 
                     out_channels=out_channels, 
                     num_layers=1, 
                     hidden_channels=hidden_channels,
                     dropout=dropout,
                     norm=norm, 
                     bias=bias, 
                     act=act, 
                     out_fn=out_fn)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            for layer in self.f.children():
                weights_init(layer, gain=gain)

    def forward(self, x, y=None): 

        out = self.conv(x)
        out = torch.flatten(out, 1)
        if y is not None: 
            out = torch.cat((out, y), dim=1)
        out = self.fc(out)
        return out
            

    def freeze_conv_layer(self): 
        for n,p in self.conv.state_dict().items(): 
            p.requires_grad = False

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)