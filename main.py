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
# imports
import argparse
import yaml
from loops import Train, Test
from tools import PseudoLabeller, Plotter
import torch
import re

# main function
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
    iterations = args.iters or 1
    path = args.path

    # Load config file
    with open(config_root, 'r') as file:
        config = yaml.safe_load(file)

    if train_flag or test_flag or plot_flag:
        # Loop over the number of loop iterations
        for i in range(iterations):
            # Modify experiment subdirectory
            #initial_sub_dir = config["logs"]["sub_dir"]
            #amended_sub_dir = "model_" + str(i)
            #config["logs"]["sub_dir"] = amended_sub_dir
            #config["logs"]["best_init"] = [float("inf"), 0]

            #if config["model"]["load_model"]:
            #    new_string = re.sub(initial_sub_dir, amended_sub_dir, config["model"]["load_model"])
            #    config["model"]["load_model"] = new_string
            #    print(new_string)
            

            with torch.cuda.device(config["loops"]["device"]):
                if train_flag:
                    trainer = Train(config)
                    trainer.train()
#
            # Execute testing
            if test_flag:
                tester = Test(config)
                tester.test()
#
            # carry out plotting
            if plot_flag:
                plotter = Plotter(config)
                plotter.plot()
    
    if label_flag:
        if not path:
            print("path not provided")
            return
        config["loops"]["device"] = "cuda:0"
        config["model"]["params"]["drop_out"] = None
        labeller = PseudoLabeller(config, path)
        labeller.label()

# excecution
if __name__ == "__main__":
    """
    Entry point of script:
        The following code defines the args parser, gathers the provided arguments, 
        them passes them to the main function
    """
    # Init parser
    parser = argparse.ArgumentParser(description="Retrieves the key parameters of training and testing models using the framework")

    # Define parsable arguments
    parser.add_argument("-config", type=str, required=True, help="Provide the path to the experiment config")
    parser.add_argument("-iters", type=int, default=1, help="Specify a number of execution iterations")
    parser.add_argument("-train", action="store_true", help="Specify if training should be executed")
    parser.add_argument("-test", action="store_true", help="Specify if testing should be executed")
    parser.add_argument("-plot", action="store_true", help="Specify if train and test results should be plotted")
    parser.add_argument("-label", action="store_true", help="Specify if labeling should be executed")
    parser.add_argument("-path", type=str, default="", help="Provide the path to the directory for the labeling task")

    # Get parsed arguments
    args = parser.parse_args()

    # Call main
    main(args)