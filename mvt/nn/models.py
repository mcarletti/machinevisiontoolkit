import os, copy

import torch
import timm

import mvt.utils
import mvt.nn.layers
import mvt.nn.utils


def get_model(name: str, input_shape: tuple, num_classes: int=None, checkpoint: str=None, strict: bool=False) -> torch.nn.Module:
    """
    Get a model instance from the model zoo.
    It also performs a fake forward pass to check if the model works properly.

    A model could be loaded with pretrained weights from the model zoo or a custom checkpoint file.
    If `checkpoint` is set to 'imagenet', the model will be loaded with the weights pretrained on ImageNet
    according to the `timm` model zoo availability. Custom models, ie. not in the `timm` model zoo, should be
    defined in the `MODEL_ZOO` dictionary and ignore the `checkpoint` parameter if it is equal to 'imagenet'.
    
    Parameters
    ----------
    `name` (str): model name or path to a custom model file
    `input_shape` (tuple): input tensor shape as tuple (C, H, W)
    `num_classes` (int): number of classes
    `checkpoint` (str): path to the checkpoint file; if 'imagenet', load the weights pretrained on ImageNet, if available
    `strict` (bool): whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's `state_dict`
    
    Return
    ------
    `model` (torch.nn.Module): model instance
    """

    # custom model architectures that are not in the timm model zoo
    MODEL_ZOO = {
        "alexnet": (AlexNet, {}),
        "cifar_cnn_pytorch": (CifarCNN_Pytorch, {}),
        "cifar_cnn_mlm": (CifarCNN_MLM, {}),
        "mnist_ffn": (MnistFFN, {}),
        "yolo": (YOLO, {"checkpoint": checkpoint}),
    }

    assert name in MODEL_ZOO or name in timm.list_models(), f"Model not found: {name}"

    if name in MODEL_ZOO:
        model_class, kwargs = MODEL_ZOO[name]
        model = model_class(input_shape=input_shape, num_classes=num_classes, **kwargs)
    else:
        model = timm.create_model(name, num_classes=num_classes, pretrained=(checkpoint == "imagenet"))

    _ = mvt.nn.utils.fake_forward_pass(model, input_shape)

    if checkpoint is not None and checkpoint != "imagenet":
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


class MnistFFN(torch.nn.Module):

    def __init__(self, input_shape: tuple=(1, 28, 28), num_classes: int=10, *args, **kwargs) -> None:
        """
        FFN model for MNIST.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        """
        super().__init__()

        act_fn = "relu"

        self.features = torch.nn.Sequential(
            mvt.nn.layers.Flatten(),
            mvt.nn.layers.Linear(input_shape[1]*input_shape[2], 128, activation=act_fn),
            mvt.nn.layers.Linear(128, 64, activation=act_fn),
        )

        self.classifier = mvt.nn.layers.Linear(64, num_classes, activation="softmax")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class YOLO(torch.nn.Module):

    def __init__(self, input_shape: tuple=(3, 224, 224), num_classes: int=80, checkpoint=None, *args, **kwargs) -> None:
        """
        YOLO model.
        
        Parameters
        ----------
        `input_shape` (tuple): input tensor shape as tuple (C, H, W)
        `num_classes` (int): number of classes
        `checkpoint` (str): path to the checkpoint file
        """
        super().__init__()

        self.backbone = timm.create_model("tf_mobilenetv3_small_minimal_100", pretrained=(checkpoint == "imagenet"), features_only=True, in_chans=input_shape[0])

        # get the number of output channels of the backbone
        # to feed the SPPF and PANet; at least two outputs are required
        fake_input = torch.zeros((1, *input_shape))
        fake_output = self.backbone(fake_input)
        assert len(fake_output) >= 2, "Error: The backbone should have at least two outputs to feed the PANet."
        bb_ch_out = fake_output[-1].shape[1]

        self.sppf = mvt.nn.layers.SPPF(bb_ch_out, 1024)

        self.panet_conv0     = mvt.nn.layers.Conv(1024, fake_output[-1].shape[1], kernel_size=1)
        self.panet_add0      = mvt.nn.layers.Add()
        self.panet_upsample0 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.panet_conv1     = mvt.nn.layers.Conv(fake_output[-1].shape[1], fake_output[-2].shape[1], kernel_size=1)
        self.panet_add1      = mvt.nn.layers.Add()
        self.panet_upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.panet_conv2     = mvt.nn.layers.Conv(fake_output[-2].shape[1], fake_output[-2].shape[1], kernel_size=1)
        self.panet_add2      = mvt.nn.layers.Add()

        self.panet_conv3     = mvt.nn.layers.Conv(fake_output[-2].shape[1], fake_output[-1].shape[1], kernel_size=3, stride=2, padding=1)
        self.panet_add3      = mvt.nn.layers.Add()

        self.panet_conv4     = mvt.nn.layers.Conv(fake_output[-1].shape[1], 1024, kernel_size=3, stride=2, padding=1)

        self.head = lambda x: x

        if num_classes > 0:

            fake_input = torch.zeros((1, *input_shape))
            fake_output = self.forward(fake_input)
            num_features = fake_output.view(-1).size(0)

            self.head = torch.nn.Sequential(
                mvt.nn.layers.Flatten(),
                torch.nn.Dropout(0.5),
                mvt.nn.layers.Linear(num_features, 128, activation="relu"),
                torch.nn.Dropout(0.5),
                mvt.nn.layers.Linear(128, num_classes, activation="softmax"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_outs = self.backbone(x)  # this populates self.features
        x00 = bb_outs[-1]

        x01 = self.sppf(x00)

        x02 = self.panet_conv0(x01)
        x03 = self.panet_add0(x02, bb_outs[-1])
        x04 = self.panet_upsample0(x03)

        x05 = self.panet_conv1(x04)
        x06 = self.panet_add1(x05, bb_outs[-2])
        x07 = self.panet_upsample1(x06)

        x08 = self.panet_conv2(x07)
        x09 = self.panet_add2(x08, x07)

        x10 = self.panet_conv3(x09)
        x11 = self.panet_add3(x10, x04)

        x12 = self.panet_conv4(x11)

        y = self.head(x12)

        return y