import functools
from abc import ABC, abstractmethod
from collections import Iterable

import torch

import mvt.nn.activations


class _Layer(ABC, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        """Base class for layers."""
        super(_Layer, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Perform forward pass on inputs"""
        pass


class Conv(_Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, groups: int=1, bias: bool=True, dilation: int=1, batch_norm: bool=True, activation: str="relu") -> None:
        """
        Convolutional layer with optional batch normalization and activation function.
        
        Parameters
        ----------
        `in_channels` (int): number of input channels
        `out_channels` (int): number of output channels, ie. number of filters
        `kernel_size` (int): kernel size (square)
        `stride` (int): stride, that is the number of pixels to move the kernel at each step
        `padding` (int): padding, that is the number of pixels to add to the input borders before applying the kernel
        `groups` (int): number of groups, that is the number of input and output channels to use in the convolution
        `bias` (bool): whether to use bias, ie. the constant term in the linear equation
        `dilation` (int): dilation factor, that is the spacing between kernel elements
        `batch_norm` (bool): whether to use batch normalization
        `activation` (str): activation function name
        """
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias, dilation=dilation)
        self.bn   = torch.nn.BatchNorm2d(out_channels) if batch_norm else lambda x: x
        self.act  = mvt.nn.activations.get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Linear(_Layer):

    def __init__(self, in_channels: int, out_channels: int, bias: bool=True, activation: str="relu") -> None:
        """
        Linear layer with optional activation function.

        Parameters
        ----------
        `in_channels` (int): number of input channels
        `out_channels` (int): number of output channels
        `bias` (bool): whether to use bias, ie. the constant term in the linear equation
        `activation` (str): activation function name
        """
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.act = mvt.nn.activations.get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class MaxPool(_Layer):

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0) -> None:
        """
        Max pooling layer.
        
        Parameters
        ----------
        `kernel_size` (int): kernel size (square)
        `stride` (int): stride, that is the number of pixels to move the kernel at each step
        `padding` (int): padding, that is the number of pixels to add to the input borders before applying the kernel
        """
        super(MaxPool, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AvgPool(_Layer):

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0) -> None:
        """
        Average pooling layer.
        
        Parameters
        ----------
        `kernel_size` (int): kernel size (square)
        `stride` (int): stride, that is the number of pixels to move the kernel at each step
        `padding` (int): padding, that is the number of pixels to add to the input borders before applying the kernel
        """
        super(AvgPool, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class GlobalAvgPool(_Layer):

    def __init__(self) -> None:
        """Global average pooling layer"""
        super(GlobalAvgPool, self).__init__()
        self.pool = functools.partial(torch.mean, dim=[2, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Flatten(_Layer):

    def __init__(self) -> None:
        """Flatten layer"""
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class Add(_Layer):

    def __init__(self) -> None:
        """Addition layer"""
        super(Add, self).__init__()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return torch.add(x0, x1)


class Concat(_Layer):

    def __init__(self):
        """Concatenation layer"""
        super(Concat, self).__init__()

    def forward(self, x: Iterable) -> torch.Tensor:
        return torch.concat(x)


class ResBlock(_Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1, groups: int=1, bias: bool=True, dilation: int=1, batch_norm=True, activation: str="relu") -> None:
        """
        Residual block with two convolutional layers and optional batch normalization and activation function.

        Parameters
        ----------
        `in_channels` (int): number of input channels
        `out_channels` (int): number of output channels, ie. number of filters
        `kernel_size` (int): kernel size (square)
        `stride` (int): stride, that is the number of pixels to move the kernel at each step
        `padding` (int): padding, that is the number of pixels to add to the input borders before applying the kernel
        `groups` (int): number of groups, that is the number of input and output channels to use in the convolution
        `bias` (bool): whether to use bias, ie. the constant term in the linear equation
        `dilation` (int): dilation factor, that is the spacing between kernel elements
        `batch_norm` (bool): whether to use batch normalization
        `activation` (str): activation function name
        """
        super().__init__()
        self.conv1 = Conv( in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, batch_norm, activation)
        self.conv2 = Conv(out_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, batch_norm, activation)

        if in_channels != out_channels:
            self.residual = Conv( in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, batch_norm, activation)
        else:
            self.residual = mvt.nn.activations.get_activation(activation, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        return self.conv2(h) + self.residual(x)


class C3(_Layer):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        C3 block from Ultralytics.

        Parameters
        ----------
        `in_channels` (int): number of input channels
        `out_channels` (int): number of output channels, ie. number of filters
        """
        super().__init__()
        hidden_channels = out_channels // 4
        self.conv1  = Conv(in_channels,     hidden_channels, kernel_size=1)
        self.conv2a = Conv(hidden_channels, hidden_channels, kernel_size=1, dilation=2)
        self.conv2b = Conv(hidden_channels, hidden_channels, kernel_size=1, dilation=4)
        self.conv2c = Conv(hidden_channels, hidden_channels, kernel_size=1, dilation=6)
        self.conv2d = Conv(hidden_channels, hidden_channels, kernel_size=1, dilation=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1  = self.conv1(x)
        h2a = self.conv2a(h1)
        h2b = self.conv2b(h1)
        h2c = self.conv2c(h1)
        h2d = self.conv2d(h1)
        h2 = torch.cat([h2a, h2b, h2c, h2d], 1)
        return x + h2


class SPPF(_Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=5) -> None:
        """
        Spatial Pyramid Pooling Fusion block with two convolutional layers from Ultralytics.

        Parameters
        ----------
        `in_channels` (int): number of input channels
        `out_channels` (int): number of output channels, ie. number of filters
        `kernel_size` (int): kernel size (square)
        """
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv(in_channels,         hidden_channels, kernel_size=1)
        self.conv2 = Conv(hidden_channels * 4, out_channels,    kernel_size=1)
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.pool(h1)
        h3 = self.pool(h2)
        h4 = self.pool(h3)
        h5 = torch.cat([h1, h2, h3, h4], 1)
        return self.conv2(h5)
