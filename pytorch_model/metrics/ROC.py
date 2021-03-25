import torch.nn as nn
from sklearn.metrics import roc_auc_score

class ROCMulti(nn.Module):
    def __init__(self, multi_class='ovr', average='macro'):
        '''
            - multi_class: 'ovr' One vs Rest, 'ovo' One vs One
            - average: averaging of the data
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        '''
        super().__init__()
        self.multi_class = multi_class
        self.average = average

    def forward(self, y_true, y_pred):
        true = y_true.numpy()
        pred = y_pred.detach().numpy()
        return roc_auc_score(true, pred, multi_class=self.multi_class, average=self.average)