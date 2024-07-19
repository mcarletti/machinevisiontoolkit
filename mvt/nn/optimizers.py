import torch


def get_optimizer(name: str, module: torch.nn.Module, learning_rate: float) -> torch.nn.Module:
    """
    Get optimizer function by name.

    Parameters
    ----------
    `name` (str): optimizer name.
    `module` (torch.nn.Module): model to optimize.
    `learning_rate` (float): learning rate.

    Returns:
    --------
    `optimizer` (torch.optim.Optimizer): optimizer instance.
    """

    OPTIMIZER_ZOO = {
        "sgd":  (torch.optim.SGD,  {"momentum": 0.9}),
        "adam": (torch.optim.Adam, {}),
    }

    assert name in OPTIMIZER_ZOO, f"Invalid optimizer name: {name}"

    opt_fn, kwargs = OPTIMIZER_ZOO.get(name)

    return opt_fn(module.parameters(), lr=learning_rate, **kwargs)


def get_scheduler(optimizer: torch.optim.Optimizer, name: str, max_epochs: int = 100, min_lr: float = 1e-8) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler by name.

    Parameters
    ----------
    `optimizer` (torch.optim.Optimizer): optimizer instance.
    `name` (str): scheduler name.
    `max_epochs` (int): maximum number of epochs; this is used as the cycle length for cosine scheduler.
    `min_lr` (float): minimum learning rate.

    Returns:
    --------
    `scheduler` (torch.optim.lr_scheduler._LRScheduler): scheduler instance.
    """

    SCHEDULER_ZOO = {
        "constant": (torch.optim.lr_scheduler.LambdaLR,          {"lr_lambda": lambda epoch: 1.0}),
        "linear":   (torch.optim.lr_scheduler.LambdaLR,          {"lr_lambda": lambda epoch: 1.0 - epoch / max_epochs}),
        "step":     (torch.optim.lr_scheduler.StepLR,            {"step_size": 30, "gamma": 0.1}),
        "plateau":  (torch.optim.lr_scheduler.ReduceLROnPlateau, {"mode": "min", "factor": 0.1, "patience": 10, "threshold": 1e-4, "min_lr": min_lr}),
        "cosine":   (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": max_epochs, "eta_min": min_lr}),
    }

    assert name in SCHEDULER_ZOO, f"Invalid scheduler name: {name}"

    scheduler_fn, kwargs = SCHEDULER_ZOO.get(name)

    return scheduler_fn(optimizer, **kwargs)