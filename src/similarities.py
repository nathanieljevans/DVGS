'''

'''
import torch 

def cosine_similarity(): 
    return torch.nn.CosineSimilarity(dim=1) 

def dot_product(): 
    return lambda A,B: torch.bmm(A.view(A.size(0), 1, A.size(1)), B.view(B.size(0), B.size(1), 1)).squeeze()

def scalar_projection(): 
    '''scalar projection of A onto B; assumes B is validation gradient (NOTE: input is B,A ...e.,g (valid, train))'''
    return lambda B,A: torch.bmm(A.view(A.size(0), 1, A.size(1)), B.view(B.size(0), B.size(1), 1)).squeeze() / torch.norm(B, p=2, dim=1)

def C_dist(L=2, unit_norm=False):
    dist = torch.nn.PairwiseDistance(p=L)
    if unit_norm: 
        return lambda x,y: -dist(x/torch.norm(x,p=L,dim=1, keepdim=True), y/torch.norm(x,p=L,dim=1, keepdim=True))
    else: 
        return lambda x,y: -dist(x,y)