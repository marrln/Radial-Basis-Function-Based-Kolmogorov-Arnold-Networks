"""
This module provides mappings from string names to PyTorch optimizer, scheduler, and criterion classes.
It enables dynamic selection and instantiation of these components based on configuration or user input.
Mappings:
- OPTIMIZERS: Dictionary mapping optimizer names (str) to their corresponding torch.optim classes.
- SCHEDULERS: Dictionary mapping scheduler names (str) to their corresponding torch.optim.lr_scheduler classes.
- CRITERIONS: Dictionary mapping loss criterion names (str) to their corresponding torch.nn loss classes.
These mappings can be extended by adding additional entries as needed.

""" 

from typing import Any, Dict
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Mapping of optimizer names to their corresponding PyTorch classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
    "Adagrad": optim.Adagrad,
    "Adamax": optim.Adamax,
    "LBFGS": optim.LBFGS,
    "ASGD": optim.ASGD,
    "Rprop": optim.Rprop,
    "NAdam": optim.NAdam,
    # Add other optimizers here as needed
}

# Mapping of schedulers names to their corresponding PyTorch classes
SCHEDULERS = {
    "StepLR": lr_scheduler.StepLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "LinearLR": lr_scheduler.LinearLR,
    # Add other optimizers here as needed
}

# Mapping of criterion names to their corresponding PyTorch classes
CRITERIONS = {
    'BCELoss' : nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "HuberLoss" : nn.HuberLoss
    # Add other criterions here as needed
}

def get_optimizer(optimizer_type: str, params: Any, optimizer_args: Dict) -> optim.Optimizer:
    """
    Returns an instantiated optimizer from OPTIMIZERS mapping.
    :param optimizer_type: Name of the optimizer (str)
    :param params: Model parameters to optimize
    :param optimizer_args: Dictionary of optimizer arguments
    :return: Instantiated optimizer
    """
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer type: '{optimizer_type}'. "
            f"Available optimizers are: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[optimizer_type](params, **optimizer_args)


def get_scheduler(sched_type: str, optimizer: optim.Optimizer, sched_args: Dict) -> lr_scheduler._LRScheduler:
    """
    Returns an instantiated scheduler from SCHEDULERS mapping.
    :param sched_type: Name of the scheduler (str)
    :param optimizer: Optimizer instance to schedule
    :param sched_args: Dictionary of scheduler arguments
    :return: Instantiated scheduler
    """
    if sched_type not in SCHEDULERS:
        raise ValueError(
            f"Unknown scheduler type: '{sched_type}'. "
            f"Available schedulers are: {list(SCHEDULERS.keys())}"
        )
    return SCHEDULERS[sched_type](optimizer, **sched_args)


def get_criterion(criterion_type: str, criterion_args: Dict) -> nn.Module:
    """
    Returns an instantiated criterion from CRITERIONS mapping.
    :param criterion_type: Name of the criterion (str)
    :param criterion_args: Dictionary of criterion arguments
    :return: Instantiated criterion
    """
    if criterion_type not in CRITERIONS:
        raise ValueError(
            f"Unknown criterion type: '{criterion_type}'. "
            f"Available criterion are: {list(CRITERIONS.keys())}"
        )
    return CRITERIONS[criterion_type](**criterion_args)
