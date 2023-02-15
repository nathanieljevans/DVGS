import torch 

class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU, out_fn=None): 
        ''''''

        super().__init__()

        seq = []
        norm_type = torch.nn.InstanceNorm1d

        # first layer 
        if in_channels is not None: 
            seq.append(torch.nn.Linear(in_channels, hidden_channels, bias=bias))
        else: 
            seq.append(torch.nn.LazyLinear(hidden_channels, bias=bias))

        seq.append(torch.nn.Dropout(dropout))
        seq.append(act())

        for l in range(num_layers - 1): 
            if norm: seq.append(norm_type(hidden_channels))
            seq.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=bias))
            seq.append(torch.nn.Dropout(dropout))
            seq.append(act())
            
        if norm: seq.append(norm_type(hidden_channels))
        seq.append(torch.nn.Linear(hidden_channels, out_channels, bias=bias))

        if out_fn is not None:
            # softmax, sigmoid, etc 
            seq.append(out_fn)

        self.f = torch.nn.Sequential(*seq)

    def forward(self, x, y=None, idx=None): 
        return self.f(x)

    def reset_parameters(self): 
        self.apply(weight_reset)

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()