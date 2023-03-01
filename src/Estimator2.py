import torch 
from NN import NN 

class Estimator2(torch.nn.Module):
    def __init__(self, xin=None, yin=None, y_cat_dim=100, num_layers=3, hidden_channels=100, norm=True, dropout=0., act=torch.nn.ReLU): 
        ''''''

        super().__init__()

        self.fx = NN(in_channels=xin, out_channels=y_cat_dim, num_layers=int(num_layers-1), 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=None)
            
        self.fy = NN(in_channels=yin, out_channels=y_cat_dim, num_layers=int(num_layers-1), 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=None)

        self.f_out  =  torch.nn.Linear(int(y_cat_dim), 1)

    def forward(self, x, y, y_val=None): 

        zx = self.fx(x) 
        zy = self.fy(y)
        z = torch.nn.functional.relu(zx + zy)   
        out = self.f_out(z) 
        return torch.sigmoid(out)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)