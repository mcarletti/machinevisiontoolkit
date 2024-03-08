import torch


def get_optimizers(name: str, module: torch.nn.Module, learning_rate: float) -> torch.nn.Module:
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