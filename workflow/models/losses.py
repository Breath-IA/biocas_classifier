import torch
import torch.nn.functional as F

def multiclass_focal_loss(inputs, targets, alpha, gamma=2.0):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss) # Prevents numerical instability
    focal_loss = alpha[targets] * (1 - pt)**gamma * ce_loss
    return focal_loss.mean()