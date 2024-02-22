"""
Module Detials:
The module contains the model class, which is used in the train and test modules. 
The model module takes the model config as an input, collects specified model in 
the config then returns the required model to either the training or testing process
when the model attribute is called 
"""
# imports
# base packages

# third party packages

# local packages
from .mask_rcnn_model import maskrcnn_resnet50_fpn
from .dual_mask_model_dev import dual_mask_resnet50_fpn
from .polite_teacher import polite_teacher_resnet50_fpn

# class
class Models():
    def __init__(self, cfg):
        """ Initialize the Models class with the configuration dictionary """
        self.cfg = cfg
        self._exctract_cfg()
        self._initialise_mapping()

    def _initialise_mapping(self):
        """ Initialize the Models class with the configuration dictionary """
        self.model_mapping = {
            "mask_rcnn":maskrcnn_resnet50_fpn,
            "dual_mask_multi_task": dual_mask_resnet50_fpn,
            "polite_teacher_mask_rcnn": polite_teacher_resnet50_fpn
        }
    
    def _exctract_cfg(self):
        """ Extracts the model name from the configuration dictionary """
        try:
            self.model_name = self.cfg["model_name"]
        except KeyError as e:
            raise KeyError(f"Missing necessary key in configuration: {e}")

    def model(self):
        """ 
        Retrieves the model class object based on the model name and initializes
        it with the configuration dictionary
        """
        if self.model_name in self.model_mapping:
            return self.model_mapping[self.model_name](self.cfg)
        else:
            raise ValueError(f"Model '{self.model_name}' is not recognised. Please check the model name or update the model mapping.")