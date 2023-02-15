import torch 
from NN import NN 

class Estimator(torch.nn.Module):
    def __init__(self, xin=None, yin=None, y_cat_dim=10, num_layers=5, hidden_channels=100, norm=True, dropout=0., act=torch.nn.ReLU, cnn=None): 
        ''''''

        super().__init__()

        self.cnn = cnn

        if cnn is None: 
            self.f1 = NN(in_channels=xin, out_channels=y_cat_dim, num_layers=int(num_layers-2), 
                            hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False, 
                                act=act, out_fn=act())

        seq = [torch.nn.Linear(y_cat_dim + yin, int(y_cat_dim), bias=False), act(), torch.nn.Linear(int(y_cat_dim), 1, bias=False)] 
                                                           
        self.f_out  =  torch.nn.Sequential(*seq)

        self.b = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x, y): 

        if self.cnn is not None: 
            z = self.cnn(x)
        else: 
            x = x.view(x.size(0), -1)
            z = self.f1(x)

        z = torch.cat((z, y), dim=1)
        out = self.f_out(z) 
        return torch.sigmoid(out + self.b)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)