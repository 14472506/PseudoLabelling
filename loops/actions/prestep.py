"""
Detials
"""
# imports
import torch

# class
class PreStep():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
    
    def action(self):
        self.action_map = {
            "mask_rcnn": self._instance_seg_action,
            "dual_mask_multi_task": self._pseudo_action,
            "polite_teacher_mask_rcnn": self._pseudo_action
        }
        return self.action_map[self.model_name]

    def _instance_seg_action(self, epoch):
        """ Detials """
        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)
    
    def _multitask_action(self, epoch):
        """ Detials """

        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)

    def _pseudo_action(self, epoch):
        """ Detials """

        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)
