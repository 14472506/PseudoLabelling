"""
Module Detials:
The losses class returns the loss from based on the provided config. 
the module also contains the AutomaticWeightedLoss class for use in
multi loss learning applications.
"""
# import
# import base packages

# import third party packages
import torch
import torch.nn.functional as F
import torch.nn as nn

# import local packages

# class
class Losses():
    def __init__(self, cfg):
        """ Initialises the Losses class """
        self.cfg = cfg
        self.model_type = self.cfg["model_name"]
    
    def loss(self):
        """ returns the selected loss when called """
        if self.model_type == "mask_rcnn":
            if self.cfg["params"]["awl"]:
                return [self._instance_seg_loss, self._AWL(5)]
            return self._instance_seg_loss
        if self.model_type == "dual_mask_multi_task":
            if self.cfg["params"]["awl"]:
                return [self._AWL(10)]
            return self._instance_seg_loss
        
    def _instance_seg_loss(self):
        """ returns null for now """
        loss = None
        return loss
    
    def _AWL(self, num_losses=3):
        """ returns the automaticaly weighted loss """
        loss = AutomaticWeightedLoss(num_losses)
        return loss

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses=3):
        """ initialises the automatically weighted loss class """
        super().__init__()
        params = torch.ones(num_losses, requires_grad=True)
        self.params = torch.nn.Parameter(params)
    
    def forward(self, *x):
        """ defines the forward pass for the learnable AWL parameters """
        losses_sum = 0
        for i, loss in enumerate(x):
            losses_sum += 0.5/(self.params[i]**2) * loss + torch.log(1+self.params[i]**2)
        return losses_sum