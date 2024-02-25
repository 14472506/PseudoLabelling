"""
===================================================================================================
Module Detials:
This module is a high level implementations of the training process for
deep learning models. The train class which is imported by the main file
uses the provided config dictionary to initialise the other modules used 
for training. To execute model training the train method is called from 
the main file.
===================================================================================================
"""
# imports ======================================================================================= #
# Base and third-party packages
import random
import torch
import numpy as np

# local packages
from models import Models
from data_handler import Loaders
from losses import Losses
from optimisers import Optimisers, Schedulers
from .logs.logs import Logs
from .actions.preloop import PreLoop
from .actions.prestep import PreStep
from .actions.step import Step
from .actions.poststep import PostStep
from .actions.postloop import PostLoop 

# class ========================================================================================= #
class Train():
    def __init__(self, cfg):
        """
        ===========================================================================================
        Initialize the main loop with given config.
        ===========================================================================================
        """
        self.cfg = cfg
        self.seed = cfg.get("loops", {}).get("seed", 42)
        self.iter_count = 0

        # set deterministic
        self._init_deterministic_settings()

        # collect config data
        self._extract_configs()
        self._initialise_model()
        self._initialise_dataloader()
        self._initialise_losses()
        self._initialise_optimiser()
        self._initialise_actions()        
        self._initialise_logs()


    def _extract_configs(self):
        """
        =========================================================================================== 
        Extract all required parameters from config 
        ===========================================================================================
        """
        # Check keys
        required_keys = ["loops", "model", "dataset", "losses", "optimiser", "logs"]
        for key in required_keys:
            if key not in self.cfg:
                raise KeyError(f"Missing necessary key in configuration: {key}")

        # Loop parameters
        self.device = self.cfg["loops"]["device"]
        self.grad_acc = self.cfg["loops"]["grad_acc"]
        self.start = self.cfg["loops"]["start"]
        self.end = self.cfg["loops"]["end"]

        # Sub configs
        self.model_cfg = self.cfg["model"]
        self.loader_cfg = self.cfg["dataset"]
        self.loss_cfg = self.cfg["losses"]
        self.optimiser_cfg = self.cfg["optimiser"]
        self.logs_cfg = self.cfg["logs"]


    def train(self):
        """
        ===========================================================================================
        Executes the main training loop based off the config dictionary
        =========================================================================================== 
        """
        self._before_loop(self.model, self.optimiser)
        
        for epoch in range(self.start, self.end):
            self._pre_step(epoch)

            self._step(self.model,
                       self.train_loader,
                       self.val_loader,
                       self.loss, 
                       self.optimiser, 
                       self.device, 
                       self.grad_acc,
                       epoch,
                       self.logs,
                       self.iter_count,
                       self.logger
                       )
              
            self._post_step(epoch, 
                            self.model, 
                            self.optimiser, 
                            self.scheduler, 
                            self.logs, 
                            self.logger)
            
        self._post_loop()


    def _initialise_model(self):
        """
        ===========================================================================================
        Initialises the model based off the provided confign ensuring the model is sent to the
        specified device for training.
        ===========================================================================================
        """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)
        #print(self.model)


    def _initialise_dataloader(self):
        """ 
        ===========================================================================================
        Initialises the required train and valudation data loaders based of the config dictionary 
        ===========================================================================================
        """
        self.train_loader, self.val_loader = Loaders(self.loader_cfg, "train").loader()


    def _initialise_losses(self):
        """ 
        ===========================================================================================
        Initialises the required loss based of the config dictionary
        ===========================================================================================
        """
        self.loss = Losses(self.loss_cfg).loss()


    def _initialise_optimiser(self):
        """
        ===========================================================================================
        Initialises the required optimiser specified in the condig in all cases. In addition if a
        scheduler is specified in the config it is initialised here. Otherwise, the sheduler
        attibute is set to None#
        ===========================================================================================
        """
        self.optimiser = Optimisers(self.optimiser_cfg, self.model, self.loss).optimiser()
        scheduler_cfg = self.optimiser_cfg.get("sched_name", None)
        self.scheduler = Schedulers(self.optimiser_cfg, self.optimiser).scheduler() if scheduler_cfg else None


    def _initialise_logs(self):
        """ 
        ===========================================================================================
        Initialises the logging for the training loop, in addition to assiging specific attributes
        to the train loop class for passing between steps in the training process.
        ===========================================================================================  
        """
        self.logger = Logs(self.logs_cfg)
        self.iter = self.logger.get_iter()
        self.logs = self.logger.get_log()
        self.logger.init_log_file(self.cfg, self.logs)


    def _initialise_actions(self):
        """ 
        ===========================================================================================
        Initialises the key steps in the training process based of the config 
        ===========================================================================================
        """
        self._before_loop = PreLoop(self.model_cfg).action()
        self._pre_step = PreStep(self.model_cfg).action()
        self._step = Step(self.model_cfg).action()
        self._post_step = PostStep(self.model_cfg).action()
        self._post_loop = PostLoop(self.model_cfg).action()
        
    
    def _init_deterministic_settings(self):
        """ 
        ===========================================================================================
        Initialises the parameters in the training loop to ensure training is as deterministic as
        possible. This areas still needs work as some non deterministic algorithms may be able to
        be made deterministic
        ===========================================================================================
        """
        # torch settings
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True, warn_only=True)

        # random settings
        random.seed(self.seed)

        # numpy settings
        np.random.seed(self.seed)