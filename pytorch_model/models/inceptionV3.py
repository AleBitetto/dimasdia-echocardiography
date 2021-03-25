import torch.nn as nn
from torchvision.models import inception_v3

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, dropout=0.5):
        super(InceptionV3, self).__init__()
        inception = inception_v3(init_weights=False, pretrained=True, aux_logits=False)
        self.model_ID = 'InceptionV3_nClass' + str(num_classes)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'dropout': dropout}
        
        
        first_layer = inception.Conv2d_1a_3x3
        if num_channels != 3:
            first_layer.conv.in_channels = num_channels
        self.Conv2d_1a_3x3 = first_layer
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
#         self.AuxLogits = inception.AuxLogits
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = inception.avgpool
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        bottleneck_features = inception.fc.in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, num_classes),
            last_layer
        )
        
        
#         modules=list(inception.children())[:-3]
#         main_body=nn.Sequential(*modules)
        
#         first_layer = main_body[0].conv
#         if num_channels != 3:
#             first_layer.in_channels = num_channels
#         main_body[0].conv = first_layer
#         self.main_body = main_body
#         self.avgpool = inception.avgpool
#         last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
#         bottleneck_features = inception.fc.in_features
#         self.fc = nn.Sequential(
#             nn.BatchNorm1d(bottleneck_features),
#             nn.Dropout(dropout),
#             nn.Linear(bottleneck_features, num_classes),
#             last_layer
#         )
        

        
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
        
        
#         x = self.main_body(x)
    
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
#         x = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class InceptionV3FeatVis(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, dropout=0.5):
        super(InceptionV3FeatVis, self).__init__()
        inception = inception_v3(init_weights=False, pretrained=True, aux_logits=False)
        self.model_ID = 'InceptionV3_nClass' + str(num_classes)
        self.model_settings = {'num_classes': num_classes,
                             'num_channels': num_channels,
                             'dropout': dropout}
        
        first_layer = inception.Conv2d_1a_3x3
        if num_channels != 3:
            first_layer.conv.in_channels = num_channels
        self.Conv2d_1a_3x3 = first_layer
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
#         self.AuxLogits = inception.AuxLogits
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = inception.avgpool
        last_layer = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        bottleneck_features = inception.fc.in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, num_classes),
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

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
#         x = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

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

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
#         x = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        return x
    
def InceptionV3FCinput(model):
    '''
    extract model without FC part
    '''
    mod = nn.Sequential(*list(model.children())[:-1])
    
    return mod