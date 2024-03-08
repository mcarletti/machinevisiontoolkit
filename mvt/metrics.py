import torch


def accuracy(outputs: torch.tensor, labels: torch.tensor) -> float:
    """
    Compute accuracy.

    Parameters
    ----------
    `outputs` (torch.tensor): model predictions as labels, ie. class indices
    `labels` (torch.tensor): ground truth labels

    Return
    ------
    float: accuracy value in range [0, 1]
    """
    total = labels.size(0)
    correct = (outputs == labels).sum().item()
    return (10000 * correct // total) / 10000


def accuracy_from_logits(outputs: torch.tensor, labels: torch.tensor) -> float:
    """
    Compute accuracy from logits.

    Parameters
    ----------
    `outputs` (torch.tensor): model predictions in the form of logits or probabilities
    `labels` (torch.tensor): ground truth labels

    Return
    ------
    float: accuracy value in range [0, 1]
    """
    _, predicted = torch.max(outputs.data, 1)
    return accuracy(predicted, labels)