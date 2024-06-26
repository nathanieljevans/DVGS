import torch 
from NN import NN 

class NNEst(torch.nn.Module):
    def __init__(self, xin, yin, out_channels, y_cat_dim, num_layers, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU, cnn=None, out_fn=None): 
        ''''''

        super().__init__()

        self.cnn = cnn

        self.f1 = NN(in_channels=xin, out_channels=y_cat_dim, num_layers=int(num_layers-2), hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=bias, act=act)
        self.f2 = NN(in_channels=yin, out_channels=y_cat_dim, num_layers=int(num_layers-2), hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=bias,act=act)

        seq = [torch.nn.Linear(y_cat_dim, int(hidden_channels/2)), 
                                                           act(), 
                                                           torch.nn.Linear(int(hidden_channels/2), 10), 
                                                           act(),
                                                           torch.nn.Linear(10, 1)] 

        if out_fn is not None: 
            seq.append(out_fn)
                                                           
        self.f_out  =  torch.nn.Sequential(*seq)

    def forward(self, x, y, idx=None): 

        if self.cnn is not None: 
            x = self.cnn(x)
        
        z = self.f1(x) + self.f2(y)
        return self.f_out(z)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)