import torch 
from NN import NN 

class LINCS_NN(torch.nn.Module):
    def __init__(self, cell_channels, cellspace, pert_channels, pertspace, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU): 
        ''''''

        super().__init__()

        self.cell_embedding = torch.nn.Embedding(num_embeddings=len(cellspace), embedding_dim=cell_channels, scale_grad_by_freq=True, dtype=torch.float32)
        self.pert_embedding = torch.nn.Embedding(num_embeddings=len(pertspace), embedding_dim=pert_channels, scale_grad_by_freq=True, dtype=torch.float32)
        self.nn = NN(in_channels=cell_channels+pert_channels + 2, out_channels=out_channels, num_layers=num_layers, hidden_channels=hidden_channels, norm=norm, dropout=dropout, bias=bias, act=act, out_fn=None)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            for layer in self.f.children():
                weights_init(layer, gain=gain)

    def forward(self, x): 
        pert_idx, cell_idx, z_time, log_conc = x
        
        z_pert = self.pert_embedding(pert_idx)
        z_cell = self.cell_embedding(cell_idx)
        #print()
        #print('z pert', z_pert.size())
        #print('z cell', z_cell.size())
        #print('time', z_time.size())
        #print('conc', log_conc.size())
        out = torch.cat((z_pert, z_cell, z_time.unsqueeze(1), log_conc.unsqueeze(1)),dim=1)
        return self.nn(out)


def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)