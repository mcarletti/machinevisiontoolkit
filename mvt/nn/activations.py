import torch


def get_activation(name: str=None, inplace: bool=True, **kwargs) -> torch.nn.Module:
    """
    Return an activation function as a torch.nn.Module.
    If `name` is `None`, return the identity lambda function.

    Parameters
    ----------
    `name` (str): name of the activation function
    `inplace` (bool): whether to perform the operation in-place
    `kwargs` (dict): additional arguments for the activation function (e.g. `negative_slope` for LeakyReLU or `dim` for Softmax)
    
    Returns
    -------
    `activation_fn` (torch.nn.Module): activation function instance
    """

    if name is None:
        return torch.nn.Sequential()

    ACTIVATION_ZOO = {
        "sigmoid":    (torch.nn.Sigmoid,     {}),
        "tanh":       (torch.nn.Tanh,        {}),
        "relu":       (torch.nn.ReLU,        {"inplace": inplace}),
        "relu6":      (torch.nn.ReLU6,       {"inplace": inplace}),
        "leaky_relu": (torch.nn.LeakyReLU,   {"inplace": inplace, "negative_slope": kwargs.get("negative_slope", 0.01)}),
        "hswish":     (torch.nn.Hardswish,   {"inplace": inplace}),
        "hsigmoid":   (torch.nn.Hardsigmoid, {"inplace": inplace}),
        "softmax":    (torch.nn.Softmax,     {"dim": kwargs.get("dim", 1)}),
    }

    assert name in ACTIVATION_ZOO, f"Unknown activation function: {name}"

    activation_fn, kwargs = ACTIVATION_ZOO.get(name)

    return activation_fn(**kwargs)