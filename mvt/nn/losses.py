import torch


def get_loss(name: str) -> torch.nn.Module:
    """
    Get loss function by name.
    
    Params:
    -------
    `name` (str): loss function name.

    Returns:
    --------
    `loss_fn` (torch.nn.Module): loss function instance.
    """

    LOSS_ZOO = {
        "categorical_cross_entropy": (torch.nn.CrossEntropyLoss, {})
    }

    assert name in LOSS_ZOO, f"Invalid loss name: {name}"

    loss_fn, kwargs = LOSS_ZOO.get(name)

    return loss_fn(**kwargs)