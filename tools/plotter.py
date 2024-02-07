"""
Module Detials:
This module uses the results from the training and test process to plot the results
of training for presentation
"""
# imports
# base packages
import os

# third party packages
from matplotlib import pyplot as plt

# local packages
from loops import Logs

# class
class Plotter():
    """ Detials """
    def __init__(self, cfg):
        self.cfg = cfg

        # extract config
        self._extract_config()

        # initialise something? what do i call this again?
        self._initialise_optimiser()
        self._initialise_logs()

    def plot(self):
        """ Detials """
        self.plotter_map = {
            "rotnet_resnet_50": self._classifier_plot,
            "jigsaw": self._classifier_plot,
            "mask_rcnn": self._instance_seg_plot,
            "rotmask_multi_task": self._multitask_plot
        }
        self.plotter_map[self.model_name]()

    def _extract_config(self):
        """ Details """
        self.model_name = self.cfg["model"]["model_name"]

        self.optimiser_cfg = self.cfg["optimiser"]
        self.logs_cfg = self.cfg["logs"]
    
    def _initialise_logs(self):
        """ Initialises the logger module for testing """
        self.logger = Logs(self.logs_cfg)

    def _initialise_optimiser(self):
        """
        Initialises step attribute for edentifying if a scheduler has been
        used in the training process. This is used for determining weather
        pre step of pre and post step models need to be evaluated.
        """
        self.step = True if self.optimiser_cfg["sched_name"] else False
    
    def _classifier_plot(self):
        """ Detials """
        print("implement classifier plotter")

    def _instance_seg_plot(self):
        """ Detials """
        log = self.logger.load_log()
        #results = self.logger.load_results()

        # extract logs
        train_loss = log["train_loss"]
        val_loss = log["val_loss"]
        epochs = log["epochs"]
        pre_best_val = log["pre_best_val"][-1]
        post_best_val =  log["post_best_val"][-1]
        pre_best_epoch = log["pre_best_epoch"][-1]
        post_best_epoch =  log["post_best_epoch"][-1]

        # Create a new figure
        plt.figure()

        # Plot training and validation loss
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')

        # Mark the best pre and post values
        plt.scatter([pre_best_epoch, post_best_epoch], [pre_best_val, post_best_val], color='red')

        # Updated annotation with best loss and epoch values
        for label, x, y, epoch, val in zip(['Pre Best', 'Post Best'], 
                                            [pre_best_epoch, post_best_epoch], 
                                            [pre_best_val, post_best_val],
                                            [pre_best_epoch, post_best_epoch],
                                            [pre_best_val, post_best_val]):
            plt.annotate(f'{label}\nEpoch: {epoch}\nLoss: {val:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

        # Add labels and legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Epochs')

        # Instead of plt.show(), use plt.savefig()
        plt.savefig(os.path.join(self.logger.result_path, "plot.png"), format='png')
        
        # Optionally, you can close the figure after saving
        plt.close()

    def _multitask_plot(self):
        """ Details """
        log = self.logger.load_log()
        #results = self.logger.load_results()

        # extract logs
        #train_loss = log["train_loss"]
        train_sup_loss = log["train_sup_loss"]
        #train_sup_ssl_loss = log["train_sup_ssl_loss"]
        #train_ssl_loss = log["train_ssl_loss"]
        #val_loss = log["val_loss"]
        val_sup_loss = log["val_sup_loss"]        
        #val_sup_ssl_loss = log["val_sup_ssl_loss"]
        #val_ssl_loss = log["val_ssl_loss"]

        train_loss = train_sup_loss
        val_loss = val_sup_loss

        epochs = log["epochs"]
        pre_best_val = log["pre_best_val"][-1]
        post_best_val =  log["post_best_val"][-1]
        pre_best_epoch = log["pre_best_epoch"][-1]
        post_best_epoch =  log["post_best_epoch"][-1]

        # Create a new figure
        plt.figure()

        # Plot training and validation loss
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')

        # Mark the best pre and post values
        plt.scatter([pre_best_epoch, post_best_epoch], [pre_best_val, post_best_val], color='red', s=1)

        # Updated annotation with best loss and epoch values
        for label, x, y, epoch, val in zip(['Pre Best', 'Post Best'], 
                                            [pre_best_epoch, post_best_epoch], 
                                            [pre_best_val, post_best_val],
                                            [pre_best_epoch, post_best_epoch],
                                            [pre_best_val, post_best_val]):
            plt.annotate(f'{label}\nEpoch: {epoch}\nLoss: {val:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

        # Add labels and legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Epochs')

        # Instead of plt.show(), use plt.savefig()
        plt.savefig(os.path.join(self.logger.result_path, "plot.png"), format='png')
        
        # Optionally, you can close the figure after saving
        plt.close()    
