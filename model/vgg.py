import torch.nn as nn
from torchvision.models import vgg19

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=22):
        """
        Extracts features from VGG19 up to the specified layer.
        Default is layer 35 (~ relu4_4).
        """
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index])
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, x):
        return self.features(x)