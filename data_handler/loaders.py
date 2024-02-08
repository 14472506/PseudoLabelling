"""
Module Detials:
The loader class in this method is called by the train and test modules. 
The loader class uses the provided config to load the required dataset and
produce a data loader for either train, validation, or test, based on the
specified type.
"""
# imports
# import base packages
import random
import copy

# import third party packages
import torch
import numpy as np

# import local packages
from .datasets.coco import COCODataset, COCO_collate_function
from .transforms.transforms import Transforms
from .transforms.transform_wrapper import wrappers

# class
class Loaders():
    def __init__(self, cfg, type=None):
        """ Initialises the Loader class using the provided config """
        self.cfg = cfg
        self.type = type
        self._extract_config()
        self._get_dataset_class()

    """ =========== supporting init methods ========== """

    def _extract_config(self):
        """ Extract the attributes from the provided config """
        try:            
            self.model_type = self.cfg["model_name"]
            self.random_seed = self.cfg["random_seed"]
            self.col_fn = self.cfg["col_fn"]
            if self.type == "train":
                self.train_bs = self.cfg["params"]["train"]["batch_size"]
                self.val_bs = self.cfg["params"]["val"]["batch_size"]
                self.train_shuffle = self.cfg["params"]["train"]["shuffle"]
                self.val_shuffle = self.cfg["params"]["val"]["shuffle"]
                self.train_workers = self.cfg["params"]["train"]["num_workers"]
                self.val_workers = self.cfg["params"]["val"]["num_workers"]
                self.train_augs = self.cfg["params"]["train"]["augmentations"]
                self.val_augs = self.cfg["params"]["val"]["augmentations"]
            else:
                self.test_bs = self.cfg["params"]["test"]["batch_size"]
                self.test_shuffle = self.cfg["params"]["test"]["shuffle"]
                self.test_workers = self.cfg["params"]["test"]["num_workers"]
                self.test_augs = self.cfg["params"]["test"]["augmentations"]
        except KeyError as e:
            raise KeyError(f"Missing necessary key in configuration: {e}")

    def _get_dataset_class(self):
        """ retirieve the dataloader based on the model type """
        dataset_selector = {
            "mask_rcnn": COCODataset,
            "dual_mask_multi_task": COCODataset,
        }
        self.dataset_class = dataset_selector[self.model_type]
    
    """ =========== loader Method ========== """

    def loader(self):
        """ Return the data loader based on the config """
        loader_selector = {
            "mask_rcnn": self._instance_loader,
            "dual_mask_multi_task": self._dual_multitask_loader
        }
        if self.type == "train":
            train_loader, val_loader = loader_selector[self.model_type]()
            return train_loader, val_loader
        else:
            test_loader = loader_selector[self.model_type]()
            return test_loader
        
    """ =========== Loader Type Methods ========== """
        
    def _instance_loader(self):
        """ creates a dataloader for the multi task instance segmentation and classifier based models """

        if self.type == "test":

            test_dataset = self.dataset_class(self.cfg, "test")

            if self.test_augs:
                transforms = Transforms(self.cfg).transforms()
                transforms_wrapper = wrappers(self.model_type)
                test_dataset = transforms_wrapper(test_dataset, transforms)
                print("test augs applied")
            
            test_loader = self._create_dataloader(test_dataset, self.test_bs, self.test_shuffle, self.test_workers, COCO_collate_function)
            return test_loader
        
        if self.type == "train":

            train_dataset = self.dataset_class(self.cfg, "train")
            val_dataset = self.dataset_class(self.cfg, "val")

            if self.train_augs:
                train_transforms = Transforms(self.cfg).transforms()
                train_transforms_wrapper = wrappers(self.model_type)
                train_dataset = train_transforms_wrapper(train_dataset, train_transforms)

                print("train augs applied")

            if self.val_augs:
                val_transforms = Transforms(self.cfg).transforms()
                val_transforms_wrapper = wrappers(self.model_type)
                val_dataset = val_transforms_wrapper(val_dataset, val_transforms)
                print("train augs applied")

            train_loader = self._create_dataloader(train_dataset, self.train_bs, self.train_shuffle, self.train_workers, COCO_collate_function)           
            val_loader = self._create_dataloader(val_dataset, self.val_bs, self.val_shuffle, self.val_workers, COCO_collate_function)
          
            
            return train_loader, val_loader 

    def _dual_multitask_loader(self):
        """ 
        this method creates two coco data loaders for training the dual mask r-cnn semi supervised
        model

        Note, currently these are identical data loaders bar the source. ie same augs etc.
        further modification may be needed in the future.
        """
        # copying dataloader config
        sup_cfg = copy.deepcopy(self.cfg)
        ssl_cfg = copy.deepcopy(self.cfg)

        # creating cfg formats for coco dataset
        sup_cfg["source"] = sup_cfg["source"][0]
        sup_cfg["params"]["train"]["dir"] = sup_cfg["params"]["train"]["dir"][0]
        sup_cfg["params"]["val"]["dir"] = sup_cfg["params"]["val"]["dir"][0]
        sup_cfg["params"]["test"]["dir"] = sup_cfg["params"]["test"]["dir"][0]
        sup_cfg["params"]["train"]["json"] = sup_cfg["params"]["train"]["json"][0]
        sup_cfg["params"]["val"]["json"] = sup_cfg["params"]["val"]["json"][0]
        sup_cfg["params"]["test"]["json"] = sup_cfg["params"]["test"]["json"][0]  

        ssl_cfg["source"] = ssl_cfg["source"][1]
        ssl_cfg["params"]["train"]["dir"] = ssl_cfg["params"]["train"]["dir"][1]
        ssl_cfg["params"]["val"]["dir"] = ssl_cfg["params"]["val"]["dir"][1]
        ssl_cfg["params"]["test"]["dir"] = ssl_cfg["params"]["test"]["dir"][1]
        ssl_cfg["params"]["train"]["json"] = ssl_cfg["params"]["train"]["json"][1]
        ssl_cfg["params"]["val"]["json"] = ssl_cfg["params"]["val"]["json"][1]
        ssl_cfg["params"]["test"]["json"] = ssl_cfg["params"]["test"]["json"][1]

        if self.type == "test":

            sup_test_dataset = self.dataset_class(sup_cfg, "test")
            ssl_test_dataset = self.dataset_class(ssl_cfg, "test")

            if self.test_augs:
                transforms = Transforms(self.cfg).transforms()
                transforms_wrapper = wrappers(self.model_type)
                sup_test_dataset = transforms_wrapper(sup_test_dataset, transforms)
                ssl_test_dataset = transforms_wrapper(ssl_test_dataset, transforms)
                print("test augs applied")
            
            sup_test_loader = self._create_dataloader(sup_test_dataset, self.test_bs[0], self.test_shuffle, self.test_workers, COCO_collate_function)
            ssl_test_loader = self._create_dataloader(ssl_test_dataset, self.test_bs[1], self.test_shuffle, self.test_workers, COCO_collate_function)
            return [sup_test_loader, ssl_test_loader]
        
        if self.type == "train":

            sup_train_dataset = self.dataset_class(sup_cfg, "train")
            sup_val_dataset = self.dataset_class(sup_cfg, "val")
            ssl_train_dataset = self.dataset_class(ssl_cfg, "train")
            ssl_val_dataset = self.dataset_class(ssl_cfg, "val")

            if self.train_augs:
                train_transforms = Transforms(self.cfg).transforms()
                train_transforms_wrapper = wrappers(self.model_type)
                sup_train_dataset = train_transforms_wrapper(sup_train_dataset, train_transforms)
                ssl_train_dataset = train_transforms_wrapper(ssl_train_dataset, train_transforms)
                print("train augs applied")

            if self.val_augs:
                val_transforms = Transforms(self.cfg).transforms()
                val_transforms_wrapper = wrappers(self.model_type)
                sup_val_dataset = val_transforms_wrapper(sup_val_dataset, val_transforms)
                ssl_val_dataset = val_transforms_wrapper(ssl_val_dataset, val_transforms)
                print("train augs applied")

            sup_train_loader = self._create_dataloader(sup_train_dataset, self.train_bs[0], self.train_shuffle, self.train_workers, COCO_collate_function)           
            sup_val_loader = self._create_dataloader(sup_val_dataset, self.val_bs[0], self.val_shuffle, self.val_workers, COCO_collate_function)
            ssl_train_loader = self._create_dataloader(ssl_train_dataset, self.train_bs[1], self.train_shuffle, self.train_workers, COCO_collate_function)           
            ssl_val_loader = self._create_dataloader(ssl_val_dataset, self.val_bs[1], self.val_shuffle, self.val_workers, COCO_collate_function)
          
            
            return [sup_train_loader, ssl_train_loader], [sup_val_loader, ssl_val_loader]  
        
    """ =========== Supporting Methods ========== """

    def _data_split(self, dataset):
        """ Splits the data of whole datasets based on a specified config split """
        # splitting data into train and test
        train_base_size = int(len(dataset)*self.train_test_split)
        test_size = len(dataset) - train_base_size
        train_base, test = torch.utils.data.random_split(dataset, [train_base_size, test_size]) 

        # just not doing this if not needed
        if self.type == "train":
            train_size = int(len(train_base)*self.train_val_split)
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
            return [train, validation]
        else:
            return [test]

    def _init_fn_worker(self, seed):
        """ ensures the seeds are in the worker init """
        np.random.seed(seed)
        random.seed(seed)

    def _create_dataloader(self, dataset, batch_size, shuffle, num_workers, collate_fn=None):
        """ Utility function to create a DataLoader from a dataset """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)  # or any other value

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=self._init_fn_worker(self.random_seed),  # Unique seed per worker.
            generator=gen,
            collate_fn=collate_fn
        )