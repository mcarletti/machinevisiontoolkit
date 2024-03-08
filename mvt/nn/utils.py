import torch


def fake_forward_pass(model: torch.nn.Module, input_shape: tuple) -> tuple:
    """
    Perform a fake forward pass to check if the model works properly.
    This is needed in order to make a deep copy of the model otherwise a runtime error will be raised:
    RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
    
    Parameters
    ----------
    `model` (torch.nn.Module): model to check
    `input_shape` (tuple): input tensor shape as tuple (C, H, W)
    
    Return
    ------
    `output_shape` (tuple): output tensor shape as tuple (C, H, W)
    """

    # aux variables
    device = next(model.parameters()).device

    # put the model into inference mode
    is_training = model.training
    model.eval()

    with torch.no_grad():
        # create dummy input tensor
        batched_shape = (1, *input_shape)
        fake_input = torch.zeros(*batched_shape, dtype=torch.float, device=device)
        # model predictions
        fake_output = model.forward(fake_input)[0]

    # restore model mode state
    if is_training:
        model.train()

    return fake_output.numpy().shape