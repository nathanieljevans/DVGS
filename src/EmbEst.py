import torch 
from NN import NN 

class EmbEst(torch.nn.Module):
    def __init__(self, num_samples, gain=1.): 
        ''''''

        super().__init__()

        self.embed = torch.nn.Embedding(num_embeddings=num_samples, embedding_dim=1) 
        self.embed.weight.data = self.embed.weight*gain

    def forward(self, x=None, y=None, idx=None): 
        
        return torch.sigmoid(self.embed(idx))