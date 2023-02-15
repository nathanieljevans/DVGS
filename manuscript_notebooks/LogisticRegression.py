import torch 

class LogisticRegression(torch.nn.Module):
    def __init__(self, in_channels, out_channels): 
        ''''''

        super().__init__()

        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            
            weights_init(self.lin, gain=gain)

    def forward(self, x): 
        x = self.lin(x)
        x = torch.softmax(x, dim=-1)
        return x
        

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)