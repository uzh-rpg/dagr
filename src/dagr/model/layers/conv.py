import torch

from torch_geometric.data import Data

from dagr.model.layers.components import BatchNormData, Linear
from dagr.model.layers.spline_conv import MySplineConv
from dagr.model.utils import shallow_copy


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, args, degree=1) -> None:
        super(ConvBlock, self).__init__()
        self.dim = args.edge_attr_dim
        self.activation = getattr(torch.nn.functional, args.activation, torch.nn.functional.elu)
        self.conv = MySplineConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 args=args,
                                 bias=False,
                                 degree=degree)

        self.norm = BatchNormData(in_channels=out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        data = self.conv(data)
        data = self.norm(data)
        data.x = self.activation(data.x)

        return data


class ConvBlockWithSkip(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, skip_in_channel: int, args) -> None:
        super(ConvBlockWithSkip, self).__init__()
        self.dim = args.edge_attr_dim

        self.conv = MySplineConv(in_channels=in_channel,
                                 out_channels=out_channel,
                                 args=args,
                                 bias=False)

        self.activation = getattr(torch.nn.functional, args.activation, torch.nn.functional.elu)
        self.norm = BatchNormData(in_channels=out_channel)

        self.lin = Linear(skip_in_channel, out_channel, bias=False)
        self.norm_skip = BatchNormData(in_channels=out_channel)

    def forward(self, data: Data, data_skip: Data):
        data = self.conv(data)

        data_skip = self.lin(data_skip)
        data_skip = self.norm_skip(data_skip)

        data = self.norm(data)
        data.x = self.activation(data.x + data_skip.x)

        return data


class Layer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, args) -> None:
        super(Layer, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels

        self.conv_block1 = ConvBlock(in_channels, out_channels, args)
        self.conv_block2 = ConvBlockWithSkip(out_channels, out_channels, in_channels, args=args)

    def forward(self, data: Data) -> torch.Tensor:
        data_skip = shallow_copy(data)
        data = self.conv_block1(data)
        output = self.conv_block2(data, data_skip)
        return output
