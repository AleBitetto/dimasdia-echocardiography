import torch.nn as nn
from torchvision.models import resnet34

class Resnet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, fc_layers=1, dropout=0.5):
        super(Resnet34, self).__init__()
        resnet = resnet34(pretrained=True)
        self.model_ID = 'Resnet34_nClass' + str(num_classes) + '_nFC' + str(fc_layers)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'fc_layers': fc_layers,
                             'dropout': dropout}
        
        first_layer = resnet.conv1
        if num_channels != 3:
            first_layer = nn.Conv2d(in_channels=num_channels,
                                    out_channels=first_layer.out_channels,
                                    kernel_size=first_layer.kernel_size,
                                    stride=first_layer.stride,
                                    padding=first_layer.padding,
                                    padding_mode=first_layer.padding_mode,
                                    dilation=first_layer.dilation,
                                    groups=first_layer.groups,
                                    bias=first_layer.bias)
        self.conv1 = first_layer
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        bottleneck_features = resnet.fc.in_features
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        fc_module = []
        fc_module.append(nn.BatchNorm1d(bottleneck_features))
        fc_module.append(nn.Dropout(dropout))
        ll=0
        for ll in range(1, fc_layers):
            fc_module.append(nn.Linear(int(bottleneck_features / 2**(ll-1)), int(bottleneck_features / 2**ll)))
            fc_module.append(nn.BatchNorm1d(int(bottleneck_features / 2**ll)))
            fc_module.append(nn.Dropout(dropout))
        fc_module.append(nn.Linear(int(bottleneck_features / 2**ll), num_classes))
        fc_module.append(last_layer)
        self.fc = nn.Sequential(*fc_module)
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Resnet34FeatVis(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, fc_layers=1, dropout=0.5):
        super(Resnet34FeatVis, self).__init__()
        resnet = resnet34(pretrained=True)
        self.model_ID = 'Resnet34_nClass' + str(num_classes) + '_nFC' + str(fc_layers)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'fc_layers': fc_layers,
                             'dropout': dropout}
        
        first_layer = resnet.conv1
        if num_channels != 3:
            first_layer = nn.Conv2d(in_channels=num_channels,
                                    out_channels=first_layer.out_channels,
                                    kernel_size=first_layer.kernel_size,
                                    stride=first_layer.stride,
                                    padding=first_layer.padding,
                                    padding_mode=first_layer.padding_mode,
                                    dilation=first_layer.dilation,
                                    groups=first_layer.groups,
                                    bias=first_layer.bias)
        self.conv1 = first_layer
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        bottleneck_features = resnet.fc.in_features
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        fc_module = []
        fc_module.append(nn.BatchNorm1d(bottleneck_features))
        fc_module.append(nn.Dropout(dropout))
        ll=0
        for ll in range(1, fc_layers):
            fc_module.append(nn.Linear(int(bottleneck_features / 2**(ll-1)), int(bottleneck_features / 2**ll)))
            fc_module.append(nn.BatchNorm1d(int(bottleneck_features / 2**ll)))
            fc_module.append(nn.Dropout(dropout))
        fc_module.append(nn.Linear(int(bottleneck_features / 2**ll), num_classes))
        fc_module.append(last_layer)
        self.fc = nn.Sequential(*fc_module)
            
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
def Resnet34FCinput(model):
    '''
    extract model without FC part
    '''
    mod = nn.Sequential(*list(model.children())[:-1])
    
    return mod