"""
Detials
"""
# imports
import torch
import torch.nn.functional as F
import gc

# class
class PostLoop():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
    
    def action(self):
        self.action_map = {
            "mask_rcnn": self._instance_seg_action,
            "dual_mask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]

    def _multitask_action(self):
        """ Detials """
        banner = "================================================================================"
        title = " Training Complete"

        print(banner)
        print(title)
        print(banner)

    def _instance_seg_action(self):
        """ Detials """
        banner = "================================================================================"
        title = " Training Complete"

        print(banner)
        print(title)
        print(banner)

