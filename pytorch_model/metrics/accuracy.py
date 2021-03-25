import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score

class Accuracy(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred):
        preds = (y_pred > self.threshold).int()
        return (preds == y_true).sum().float() / len(preds)
    
class AccuracyMulti(nn.Module):
    def __init__(self, ordered_class_labels=[], mode='max'):
        '''
            - ordered_class_labels: list of sorted target classes
            - mode: how to set predicted class given the probabilities array y_pred. 'max': takes highest probability
        '''
        super().__init__()
        self.ordered_class_labels = ordered_class_labels
        self.mode = mode

    def forward(self, y_true, y_pred):
        
        if y_pred.shape[1] != len(self.ordered_class_labels):
            raise ValueError('y_pred shape doesn\'t match number of target classes in AccuracyMulti')
        
        if self.mode == 'max':
            _, max_ind = y_pred.max(dim=1)
            preds = torch.tensor([self.ordered_class_labels[ind] for ind in max_ind]).int()
        return (preds == y_true).sum().float() / len(preds)
    
