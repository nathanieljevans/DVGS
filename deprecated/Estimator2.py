import torch 
from NN import NN 

class Estimator2(torch.nn.Module):
    def __init__(self, xin=None, yin=None, y_cat_dim=100, num_layers=3, hidden_channels=100, norm=True, dropout=0., act=torch.nn.ReLU, out_fn=None): 
        ''''''

        self.out_fn = out_fn

        super().__init__()

        self.fx = NN(in_channels=xin, out_channels=y_cat_dim, num_layers=int(num_layers-2), 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=None)
            
        self.fy = NN(in_channels=yin, out_channels=y_cat_dim, num_layers=int(num_layers-2), 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=None)

        self.f_out  = NN(in_channels=2*y_cat_dim, out_channels=1, num_layers=2, 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=None)


    def forward(self, x, y, y_val=None): 

        zx = self.fx(x) 
        zy = self.fy(y)
        z = torch.nn.functional.relu(torch.cat((zx,zy), dim=-1))  
        out = self.f_out(z) 
        if self.out_fn is not None: 
            out = self.out_fn(out)
        return out
        

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)