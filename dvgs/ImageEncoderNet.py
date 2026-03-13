

import torch 
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet(name): 

    if name == 'resnet50': 
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif name == 'resnet18': 
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif name == 'inception': 
        #return torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        return torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.DEFAULT")

class ImageEncoderNet(torch.nn.Module):
    def __init__(self, name, out_channels, dropout=0., freeze=False): 
        ''''''
        super().__init__()

        self.name = name
        self.out_channels = out_channels
        self.resnet = get_resnet(self.name)

        if freeze: 
            for p in self.resnet.parameters(): 
                p.requires_grad_ = False

        # just removes the last linear layer
        self.resnet.fc = Null()

        if out_channels is not None: 
            self.fc = torch.nn.LazyLinear(out_channels)
            self.do = torch.nn.Dropout(dropout) 
        else: 
            self.fc = None
            self.do = None

    def reset_parameters(self): 
        device = next(self.parameters()).device
        self.resnet = get_resnet(self.name).to(device) 
        self.fc = torch.nn.LazyLinear(self.out_channels).to(device) 

    def forward(self, x):
        self.resnet.eval() # override data val methods - batchnorm doesn't work with vmap 
        out = self.resnet(x)

        if self.fc is not None: 
            out = self.do(out)
            out = self.fc(out)

        return out

class Null(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
    def forward(self, x): 
        return x