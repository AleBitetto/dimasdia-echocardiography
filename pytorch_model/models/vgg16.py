import torch.nn as nn
from torchvision.models import vgg16_bn

class VGG16(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, dropout=0.5):
        super(VGG16, self).__init__()
        vgg = vgg16_bn(pretrained=True)
        self.model_ID = 'VGG16_nClass' + str(num_classes)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'dropout': dropout}
        
        first_layer = vgg.features[0]
        if num_channels != 3:
            first_layer.in_channels = num_channels
        vgg.features[0] = first_layer
        self.feature = vgg.features
        self.avgpool = vgg.avgpool
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        bottleneck_features = vgg.classifier[0].in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
            last_layer
        )
        
    def forward(self, x):
        # mean = MEAN
        # std = STD
        x = x / 255.
        # x = torch.cat([
        #     (x[:, [0]] - mean[0]) / std[0],
        #     (x[:, [1]] - mean[1]) / std[1],
        #     (x[:, [2]] - mean[2]) / std[2],
        #     (x[:, [3]] - mean[3]) / std[3],
        # ], 1)
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class VGG16FeatVis(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, dropout=0.5):
        super(VGG16FeatVis, self).__init__()
        vgg = vgg16_bn(pretrained=True)
        self.model_ID = 'VGG16_nClass' + str(num_classes)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'dropout': dropout}
        
        first_layer = vgg.features[0]
        if num_channels != 3:
            first_layer.in_channels = num_channels
        vgg.features[0] = first_layer
        self.feature = vgg.features
        self.avgpool = vgg.avgpool
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        bottleneck_features = vgg.classifier[0].in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
            last_layer
        )
            
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # mean = MEAN
        # std = STD
        x = x / 255.
        # x = torch.cat([
        #     (x[:, [0]] - mean[0]) / std[0],
        #     (x[:, [1]] - mean[1]) / std[1],
        #     (x[:, [2]] - mean[2]) / std[2],
        #     (x[:, [3]] - mean[3]) / std[3],
        # ], 1)
        x = self.feature(x)
        
        # register the hook
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction - put all layers before hook
    def get_activations(self, x):
        
        x = x / 255.

        x = self.feature(x)
        
        return x
    
def VGG16FCinput(model):
    '''
    extract model without FC part
    '''
    mod = nn.Sequential(*list(model.children())[:-1])
    
    return mod