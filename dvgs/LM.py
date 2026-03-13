import torch 

class LM(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  bias=True): 
        ''''''

        super().__init__()

        self.f = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, y=None, idx=None): 
        return self.f(x)

    def reset_parameters(self): 
        self.apply(weight_reset)

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()