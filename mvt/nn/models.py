import os, copy

import torch

import mvt.utils
import mvt.nn.layers
import mvt.nn.utils


def get_model(name: str, input_shape: tuple, num_classes: int=None, checkpoint: str=None, strict: bool=False) -> torch.nn.Module:
    """
    Get a model instance from the model zoo.
    It also performs a fake forward pass to check if the model works properly.
    
    Parameters
    ----------
    `name` (str): model name or path to a custom model file
    `input_shape` (tuple): input tensor shape as tuple (C, H, W)
    `num_classes` (int): number of classes
    
    Return
    ------
    `model` (torch.nn.Module): model instance
    """

    MODEL_ZOO = {
        "alexnet": AlexNet,
        "resnet18": ResNet18,
        "cifar_cnn_pytorch": CifarCNN_Pytorch,
        "cifar_cnn_mlm": CifarCNN_MLM,
    }

    kwargs = {
        "input_shape": input_shape,
        "num_classes": num_classes,
    }

    assert name in MODEL_ZOO or os.path.isfile(name), f"Model not found: {name}"
    model = MODEL_ZOO[name](**kwargs)

    _ = mvt.nn.utils.fake_forward_pass(model, input_shape)

    if checkpoint is not None:
        model = load_model_weights(model, checkpoint, strict)

    return model


def load_model_weights(model: torch.nn.Module, checkpoint: str, strict: bool=False) -> torch.nn.Module:
    """
    Load the weights of a model from a checkpoint.

    Parameters
    ----------
    `model` (torch.nn.Module): model instance
    `checkpoint` (str): path to the checkpoint file
    `strict` (bool): whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's `state_dict`

    Return
    ------
    `model` (torch.nn.Module): model with loaded weights
    """

    # sanity check
    assert os.path.isfile(checkpoint), f"Checkpoint file not found: {checkpoint}"

    # load checkpoint to CPU to avoid CUDA memory leak
    source_state_dict = torch.load(checkpoint, map_location="cpu")
    target_state_dict = model.state_dict()
    common_state_dict = mvt.utils.intersect_dicts(source_state_dict, target_state_dict)

    if strict:
        assert len(common_state_dict) == len(target_state_dict), "Error: The number of parameters in the checkpoint does not match the number of parameters in the model. Try setting `strict=False`."

    model.load_state_dict(common_state_dict, strict=strict)
    print(f"Loaded {len(common_state_dict)}/{len(target_state_dict)} sets of parameters from checkpoint.")

    return model


class AlexNet(torch.nn.Module):

    def __init__(self, input_shape: tuple=(3, 224, 224), num_classes: int=1000, no_top: bool=False, *args, **kwargs) -> None:
        """
        AlexNet model.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        `no_top` (bool): whether to include the top layers
        """
        super().__init__()

        use_bn = True
        act_fn = "relu"

        self.features = torch.nn.Sequential(
            mvt.nn.layers.Conv(input_shape[0], 96, kernel_size=11, stride=4, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=3, stride=2),
            mvt.nn.layers.Conv(96, 256, kernel_size=5, padding=2, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=3, stride=2),
            mvt.nn.layers.Conv(256, 384, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.Conv(384, 384, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.Conv(384, 256, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=3, stride=2),
        )

        num_features = self.features(torch.zeros((1, *input_shape))).view(-1).size(0)

        if no_top:
            self.classifier = lambda x: x
        else:
            self.classifier = torch.nn.Sequential(
                mvt.nn.layers.Flatten(),
                mvt.nn.layers.Linear(num_features, 4096, activation=act_fn),
                torch.nn.Dropout(),
                mvt.nn.layers.Linear(4096, 4096, activation=act_fn),
                torch.nn.Dropout(),
                mvt.nn.layers.Linear(4096, num_classes, activation="softmax"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18(torch.nn.Module):

    def __init__(self, input_shape: tuple=(3, 224, 224), num_classes: int=1000, no_top: bool=False, *args, **kwargs) -> None:
        """
        ResNet-18 model.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        `no_top` (bool): whether to include the top layers
        """
        super().__init__()

        use_bn = True
        act_fn = "relu"

        self.features = torch.nn.Sequential(
            mvt.nn.layers.Conv(input_shape[0], 64, kernel_size=7, stride=2, padding=3, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=3, stride=2),
            mvt.nn.layers.ResBlockBasic( 64,  64, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic( 64,  64, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic( 64, 128, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic(128, 128, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic(128, 256, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic(256, 256, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic(256, 512, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.ResBlockBasic(512, 512, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.GlobalAvgPool(),
        )

        num_features = self.features(torch.zeros((1, *input_shape))).view(-1).size(0)

        if no_top:
            self.classifier = lambda x: x
        else:
            self.classifier = torch.nn.Sequential(
                mvt.nn.layers.Flatten(),
                mvt.nn.layers.Linear(num_features, num_classes, activation="softmax"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class CifarCNN_Pytorch(torch.nn.Module):
    # pytorch.org
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self, input_shape: tuple=(3, 32, 32), num_classes: int=10, no_top: bool=False, *args, **kwargs) -> None:
        """
        CifarCNN model.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        `no_top` (bool): whether to include the top layers
        """
        super().__init__()

        use_bn = False
        act_fn = "relu"

        self.features = torch.nn.Sequential(
            mvt.nn.layers.Conv(input_shape[0], 6, kernel_size=5, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=2),
            mvt.nn.layers.Conv(6, 16, kernel_size=5, padding=1, batch_norm=use_bn, activation=act_fn),
        )

        num_features = self.features(torch.zeros((1, *input_shape))).view(-1).size(0)

        if no_top:
            self.classifier = lambda x: x
        else:
            self.classifier = torch.nn.Sequential(
                mvt.nn.layers.Flatten(),
                mvt.nn.layers.Linear(num_features, 120, activation=act_fn),
                mvt.nn.layers.Linear(120, 84, activation=act_fn),
                mvt.nn.layers.Linear(84, num_classes, activation="softmax"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class CifarCNN_MLM(torch.nn.Module):
    # machinelearningmastery.com
    # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

    def __init__(self, input_shape: tuple=(3, 32, 32), num_classes: int=10, no_top: bool=False, *args, **kwargs) -> None:
        """
        CifarCNN model.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        `no_top` (bool): whether to include the top layers
        """
        super().__init__()

        use_bn = True
        act_fn = "relu"

        self.features = torch.nn.Sequential(
            mvt.nn.layers.Conv(input_shape[0], 32, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.Conv(32, 32, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=2),
            torch.nn.Dropout(0.2),
            mvt.nn.layers.Conv(32, 64, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.Conv(64, 64, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=2),
            torch.nn.Dropout(0.2),
            mvt.nn.layers.Conv( 64, 128, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.Conv(128, 128, kernel_size=3, padding=1, batch_norm=use_bn, activation=act_fn),
            mvt.nn.layers.MaxPool(kernel_size=2),
        )

        num_features = self.features(torch.zeros((1, *input_shape))).view(-1).size(0)

        if no_top:
            self.classifier = lambda x: x
        else:
            self.classifier = torch.nn.Sequential(
                mvt.nn.layers.Flatten(),
                torch.nn.Dropout(0.5),
                mvt.nn.layers.Linear(num_features, 128, activation=act_fn),
                torch.nn.Dropout(0.5),
                mvt.nn.layers.Linear(128, num_classes, activation="softmax"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x