import torch 
from NN import NN 

class LincsEmbNN(torch.nn.Module):
    def __init__(self, cell_channels, num_lines, pert_channels, num_perts, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True, act=torch.nn.ReLU): 
        ''''''
        super().__init__()

        self.cell_embedding = torch.nn.Embedding(num_embeddings=num_lines, embedding_dim=cell_channels, scale_grad_by_freq=True, dtype=torch.float32)
        self.pert_embedding = torch.nn.Embedding(num_embeddings=num_perts, embedding_dim=pert_channels, scale_grad_by_freq=True, dtype=torch.float32)
        self.nn = NN(in_channels=cell_channels+pert_channels + 2, out_channels=out_channels, num_layers=num_layers, hidden_channels=hidden_channels, norm=norm, dropout=dropout, bias=bias, act=act, out_fn=None)

    def reset_parameters(self, gain=1): 
        with torch.no_grad(): 
            for layer in self.f.children():
                weights_init(layer, gain=gain)

    def forward(self, pert_idx, cell_idx, z_time, log_conc): 
        z_pert = self.pert_embedding(pert_idx)
        z_cell = self.cell_embedding(cell_idx)
        out = torch.cat((z_pert, z_cell, z_time.unsqueeze(1), log_conc.unsqueeze(1)), dim=1)
        return self.nn(out)

def weights_init(m, gain=1):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)