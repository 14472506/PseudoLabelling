"""
Detials
"""
# imports
import torch

# class
class PreLoop():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
        self.load_model = self.cfg["load_model"]
        self.device = self.cfg["device"]
    
    def action(self):
        self.action_map = {
            "mask_rcnn": self._instance_seg_action,
            "dual_mask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]
    
    def _instance_seg_action(self, model, optimiser):
        """ Details """
        banner = "================================================================================"
        title = " Instance Seg Training "

        if self.load_model:
            # Load model weights here.
            checkpoint = torch.load(self.load_model, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])

            for state in optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            print("model_loaded")
            
        print(banner)
        print(title)
        print(banner)
    
    def _multitask_action(self, model, optimiser):
        """ Detials """
        banner = "================================================================================"
        title = " Multi Task Training "

        if self.load_model:
            # Load model weights here.
            checkpoint = torch.load(self.load_model, map_location=self.device)
            pretrained_dict = checkpoint["state_dict"]
            model_state_dict = model.state_dict()

            pretrained_state_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
            model_state_dict.update(pretrained_state_dict)

            model.load_state_dict(model_state_dict)

            for state in optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            print("model_loaded")
            
        print(banner)
        print(title)
        print(banner)

        banner = "================================================================================"
        title = " Multi Task Training "

        print(banner)
        print(title)
        print(banner)