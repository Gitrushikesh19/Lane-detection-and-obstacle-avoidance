import torch
import torch.nn as tnn

class DiceLoss(tnn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        batch_dice = []
        for i in range(y_pred.shape[0]):
            y_pred = y_pred[i].flatten()
            y_true = y_true[i].flatten()
            intersection = (y_pred*y_true).sum()
            Dice_i = (2.0*intersection + self.smooth)/(y_pred.sum() + y_true.sum() + self.smooth)
            batch_dice.append(1-Dice_i)
        return torch.stack(batch_dice).mean()

class BCEDiceLoss(tnn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = tnn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        bce_dice = self.bce(y_pred, y_true) + self.dice(y_pred, y_true)
        return bce_dice
