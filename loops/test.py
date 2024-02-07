"""
Module Detials:
This module is a high level implementations of the testing process for
deep learning models. The test class which is imported by the main file
uses the provided config dictionary to initialise the other modules used 
for testing. To execute model testing the test method is called from 
the main file.
"""
# imports
# base packages

# third party packages

# local packages
from models import Models
from data_handling import Loaders
from losses import Losses
from .logs.logs import Logs
from .actions.test import TestAction

# class
class Test():
    def __init__(self, cfg):
        """ Initialises the Test class with the provided config """
        self.cfg = cfg
        
        # extract configs
        self._extract_cfg()

        # initialise key test components
        self._initialise_model()
        self._initialise_dataloader()
        self._initialise_optimiser()
        self._initialise_actions()
        self._initialise_logs()

    def test(self):
        """ Test method called to execute testing based on setup in the config """
        self.test_action(self.model, self.test_loader, self.step, self.logger, self.device)
    
    def _extract_cfg(self):
        """ Extracts all key paramaters from the provided config dictionary """
        try:
            self.device = self.cfg["loops"]["device"]
            self.model_cfg = self.cfg["model"]
            self.optimiser_cfg = self.cfg["optimiser"]
            self.logs_cfg = self.cfg["logs"]
            self.loader_cfg = self.cfg["dataset"]
        except KeyError as e:
            raise KeyError(f"Missing necessary key in configuration: {e}")

    def _initialise_model(self):
        """
        Initialises the model based off the provided confign
        ensuring the model is sent to the specified device for
        testing
        """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)

    def _initialise_dataloader(self):
        """ 
        Initialises the required test data loaders based of the config
        dictionary 
        """
        self.test_loader = Loaders(self.loader_cfg, "test").loader()

    def _initialise_optimiser(self):
        """
        Initialises step attribute for edentifying if a scheduler has been
        used in the training process. This is used for determining weather
        pre step of pre and post step models need to be evaluated.
        """
        self.step = True if self.optimiser_cfg["sched_name"] else False

    def _initialise_logs(self):
        """ Initialises the logger module for testing """
        self.logger = Logs(self.logs_cfg)
    
    def _initialise_actions(self):
        """ Initialising the testing action to be carry out model testing """
        self.test_action = TestAction(self.model_cfg).action()