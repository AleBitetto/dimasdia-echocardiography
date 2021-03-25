import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=1, size_average=True):
#         super(FocalLoss2, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input).long()
#         print('\n1', logpt.type())
#         print('\n11', target.type())
#         logpt = logpt.gather(1,target.long())
#         print('\n2', logpt.type())
#         logpt = logpt.view(-1)
#         print('\n3', logpt.data.exp().type())
#         pt = Variable(logpt.data.exp().float().cuda())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
      
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, eval_logits=False, reduce=True, multi_class=False):
        '''
        - eval_logits: evaluate sigmoid or softmax to raw prediction
        
        for multiclass problem prediction output is supposed to be the softmax (so eval_logits=False),
        so only log()+NLLLoss will be applied
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eval_logits = eval_logits
        self.reduce = reduce
        self.multi_class = multi_class

    def forward(self, inputs, targets):
        if self.multi_class:
            if self.eval_logits:
                loss = F.cross_entropy(inputs, targets, reduction='none') # apply only logSoftmax and then NLLLoss
            else:
                loss = F.nll_loss(torch.log(inputs), targets, reduction='none') # apply log and then NLLLoss
        else:
            if self.eval_logits:
                loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # apply sigmoid and then log
            else:
                loss = F.binary_cross_entropy(inputs, targets, reduction='none') # apply only log
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss