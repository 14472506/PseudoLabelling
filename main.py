"""
Script Detials:
This is a top level script for executing the training and testing of deep learning models
for specified classifier, instance segmentation, and multi task model training.

Usage: 
    - Specify the path to the experiment config file
    - Set the number of iterations by which the experiment should be executed
    - Specify weather training should take place
    - Specify weather testing should take place
"""
# imports ======================================================================================= #
import argparse
import yaml
from loops import Train, Test
from tools import PseudoLabeller, Plotter
import torch

# classes and functions ========================================================================= #
def load_config(config_root):
    """ loads config file from condig root"""
    with open(config_root, 'r') as file:
        return yaml.safe_load(file)

class TaskExecutor():
    """ handles the execution of different tasks """
    def __init__(self, config):
        self.config = config
    
    def train(self, start, iters):
        """ handles the training of models based on provided config """
        for i in range(start, iters):
            self.config["logs"]["sub_dir"] = "model_" + str(i)
            self.config["logs"]["best_init"] = [float("inf"), 0]
            with torch.cuda.device(self.config["loops"]["device"]):
                trainer = Train(self.config)
                trainer.train()

    def test(self, start, iters):
        """ handle model testing for trained models based on provided config """
        for i in range(start, iters):
            self.config["logs"]["sub_dir"] = "model_" + str(i)
            with torch.cuda.device(self.config["loops"]["device"]):
                tester = Test(self.config)
                tester.test()

    def plot(self, start, iters):
        """ handles plotting tasks based on config"""
        for i in range(start, iters):
            self.config["logs"]["sub_dir"] = "model_" + str(i)
            plotter = Plotter(self.config)
            plotter.plot()

    def label(self, path):
        """ handles the labelling process """
        if not path:
            print("path not provided")
            return
        self.config["model"]["params"]["drop_out"] = None
        labeller = PseudoLabeller(self.config, path)
        labeller.label()

def main(args):
    """
    Main function for executing the experimental training and desting of 
    deep learning models.
    Args: 
        - (argparser.Namespace): parsing the command line
    """
    # Extract args
    config_root = args.config
    train_flag = args.train
    test_flag = args.test
    label_flag = args.label
    plot_flag = args.plot 
    start = args.start or 0
    iterations = args.iters or 1
    path = args.path

    config = load_config(config_root)
    executor = TaskExecutor(config)

    if train_flag:
        executor.train(start, iterations)
    
    if test_flag:
        executor.test(start, iterations)

    if plot_flag:
        executor.plot(start, iterations)
    
    if label_flag:
        executor.label(path)

# excecution ==================================================================================== #
if __name__ == "__main__":
    """
    Entry point of script:
        The following code defines the args parser, gathers the provided arguments, 
        them passes them to the main function
    """
    # Init parser
    parser = argparse.ArgumentParser(description="Retrieves the key parameters of training and testing models using the framework")

    # All cases
    parser.add_argument("-config", type=str, required=True, help="Provide the path to the experiment config")

    # train and test
    parser.add_argument("-start", type=int, default=0, help="start point default 0 otherwise specifified")
    parser.add_argument("-iters", type=int, default=1, help="Specify a number of execution iterations")
    parser.add_argument("-train", action="store_true", help="Specify if training should be executed")
    parser.add_argument("-test", action="store_true", help="Specify if testing should be executed")

    # plotting
    parser.add_argument("-plot", action="store_true", help="Specify if train and test results should be plotted")

    # labelling
    parser.add_argument("-label", action="store_true", help="Specify if labeling should be executed")
    parser.add_argument("-path", type=str, default="", help="Provide the path to the directory for the labeling task")

    # Get parsed arguments
    args = parser.parse_args()

    # Call main
    main(args)