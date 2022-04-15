import torch
import numpy as np

def dice_loss_org_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat, iflat), weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))
    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_org(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_II_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat/(iflat+delta), tflat),weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)),weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
    A_sum = torch.sum(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_II(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat/(iflat+delta), tflat))

    A_sum = torch.sum(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_III_weights(pred, target, weights, alpha=2):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(torch.pow(iflat, alpha), tflat),weights_flat))

    A_sum = torch.sum(torch.mul(torch.mul(torch.pow(iflat, alpha), torch.pow(iflat, alpha)),weights_flat))
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_III(pred, target, alpha=2):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.pow(iflat, alpha), tflat))

    A_sum = torch.sum(torch.mul(torch.pow(iflat, alpha), torch.pow(iflat, alpha)))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return (intersection) / (A_sum + B_sum + 0.0001)