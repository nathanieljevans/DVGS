import torch 

class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU, out_fn=None): 
        ''''''

        super().__init__()

        seq = []

        # first layer 
        seq.append(torch.nn.Linear(in_channels, hidden_channels, bias=bias))
        seq.append(torch.nn.Dropout(dropout))
        seq.append(act())

        for l in range(num_layers - 1): 
            if norm: seq.append(torch.nn.BatchNorm1d(hidden_channels))
            seq.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=bias))
            seq.append(torch.nn.Dropout(dropout))
            seq.append(act())
            
        # output layer
        if norm: seq.append(torch.nn.BatchNorm1d(hidden_channels))
        seq.append(torch.nn.Linear(hidden_channels, 10, bias=bias))
        seq.append(torch.nn.Linear(10, out_channels, bias=bias))

        if out_fn is not None:
            # softmax, sigmoid, etc 
            seq.append(out_fn())

        self.f = torch.nn.Sequential(*seq)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            for layer in self.f.children():
                weights_init(layer, gain=gain)

    def forward(self, x): 
        return self.f(x)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)