import torch

import torch_geometric.transforms as T

from torch_geometric.data import Data
from dagr.model.layers.ev_tgn import EV_TGN
from dagr.model.layers.pooling import Pooling
from dagr.model.layers.conv import Layer
from dagr.model.layers.components import Cartesian
from dagr.model.networks.net_img import HookModule
from dagr.model.utils import shallow_copy
from torchvision.models import resnet18, resnet34, resnet50


def sampling_skip(data, image_feat):
    image_feat_at_nodes = sample_features(data, image_feat)
    return torch.cat((data.x, image_feat_at_nodes), dim=1)

def compute_pooling_at_each_layer(pooling_dim_at_output, num_layers):
    py, px = map(int, pooling_dim_at_output.split("x"))
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    poolings = []
    for i in range(num_layers):
        pooling = pooling_base / 2 ** (3 - i)
        pooling[-1] = 1
        poolings.append(pooling)
    poolings = torch.stack(poolings)
    return poolings


def events_to_histogram(p, b, x, y, shape):
    batch_size, channels, height, width = shape
    x = x * width
    y = y * height
    p_01 = (p + 1) // 2
    histogram = torch.zeros((batch_size, channels, height, width), device=p.device, dtype=torch.float)
    idx = b.long() * height * width * channels + p_01.long() * height * width + y.long() * width + x.long()
    histogram.put_(index=idx, source=torch.ones_like(p_01), accumulate=True)
    return histogram

class Net(torch.nn.Module):
    def __init__(self, args, height, width):
        super().__init__()

        channels = [1, int(args.base_width*32), int(args.after_pool_width*64),
                    int(args.net_stem_width*128),
                    int(args.net_stem_width*128),
                    int(args.net_stem_width*128)]

        self.out_channels_cnn = []
        if args.use_image:
            img_net = eval(args.img_net)
            self.out_channels_cnn = [256, 256]
            input_channels = 6 if args.train_ev_yolox else 3
            self.net = HookModule(img_net(pretrained=True),
                                  input_channels=input_channels,
                                  height=height, width=width,
                                  feature_layers=["conv1", "layer1", "layer2", "layer3", "layer4"],
                                  output_layers=["layer3", "layer4"],
                                  feature_channels=channels[1:],
                                  output_channels=self.out_channels_cnn)

        self.use_image = args.use_image
        self.num_scales = args.num_scales
        self.train_ev_yolox = args.train_ev_yolox

        self.num_classes = dict(dsec=2, ncaltech101=100).get(args.dataset, 2)

        self.events_to_graph = EV_TGN(args)

        output_channels = channels[1:]
        self.out_channels = output_channels[-2:]

        input_channels = channels[:-1]
        if self.use_image:
            input_channels = [input_channels[i] + self.net.feature_channels[i] for i in range(len(input_channels))]

        # parse x and y pooling dimensions at output
        poolings = compute_pooling_at_each_layer(args.pooling_dim_at_output, num_layers=4)
        max_vals_for_cartesian = 2*poolings[:,:2].max(-1).values
        self.strides = torch.ceil(poolings[-2:,1] * height).numpy().astype("int32").tolist()
        self.strides = self.strides[-self.num_scales:]

        effective_radius = 2*float(int(args.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        self.conv_block1 = Layer(2+input_channels[0], output_channels[0], args=args)

        cart1 = T.Cartesian(norm=True, cat=False, max_value=2*effective_radius)
        self.pool1 = Pooling(poolings[0], width=width, height=height, batch_size=args.batch_size,
                             transform=cart1, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer2 = Layer(input_channels[1]+2, output_channels[1], args=args)

        cart2 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[1])
        self.pool2 = Pooling(poolings[1], width=width, height=height, batch_size=args.batch_size,
                             transform=cart2, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer3 = Layer(input_channels[2]+2, output_channels[2],  args=args)

        cart3 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[2])
        self.pool3 = Pooling(poolings[2], width=width, height=height, batch_size=args.batch_size,
                             transform=cart3, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer4 = Layer(input_channels[3]+2, output_channels[3],  args=args)

        cart4 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[3])
        self.pool4 = Pooling(poolings[3], width=width, height=height, batch_size=args.batch_size,
                             transform=cart4, aggr='mean', keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer5 = Layer(input_channels[4]+2, output_channels[4],  args=args)

        self.cache = []

    def get_output_sizes(self):
        poolings = [self.pool3.voxel_size[:2], self.pool4.voxel_size[:2]]
        output_sizes = [(1 / p + 1e-3).cpu().int().numpy().tolist()[::-1] for p in poolings]
        return output_sizes

    def forward(self, data: Data, reset=True):
        if self.use_image:
            image = data.image
            if self.train_ev_yolox:
                histogram = events_to_histogram(p=data.x[:,0], x=data.pos[:,0], y=data.pos[:,1],
                                                shape=image.shape, b=data.batch)
                image = torch.cat([image, histogram], dim=1)
            image_feat, image_outputs = self.net(image)

        if hasattr(data, 'reset'):
            reset = data.reset

        data = self.events_to_graph(data, reset=reset)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[0].detach())
            data.skipped = True
            data.num_image_channels = image_feat[0].shape[1]

        data = self.edge_attrs(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.conv_block1(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[1].detach())

        data = self.pool1(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[1].shape[1]

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer2(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[2].detach())

        data = self.pool2(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[2].shape[1]

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer3(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[3].detach())

        data = self.pool3(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[3].shape[1]

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer4(data)

        out3 = shallow_copy(data)
        out3.pooling = self.pool3.voxel_size[:3]

        if self.use_image:
            data.x = sampling_skip(data, image_feat[4].detach())

        data = self.pool4(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[4].shape[1]

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer5(data)

        out4 = data
        out4.pooling = self.pool4.voxel_size[:3]

        output = [out3, out4]

        if self.use_image:
            return output[-self.num_scales:], image_outputs[-self.num_scales:]
        return output[-self.num_scales:]


def sample_features(data, image_feat, image_sample_mode="bilinear"):
    if data.batch is None or len(data.batch) != len(data.pos):
        data.batch = torch.zeros(len(data.pos), dtype=torch.long, device=data.x.device)
    return _sample_features(data.pos[:,0] * data.width[0],
                            data.pos[:,1] * data.height[0],
                            data.batch.float(), image_feat,
                            data.width[0],
                            data.height[0],
                            image_feat.shape[0],
                            image_sample_mode)

def _sample_features(x, y, b, image_feat, width, height, batch_size, image_sample_mode):
    x = 2 * x / (width - 1) - 1
    y = 2 * y / (height - 1) - 1

    batch_size = batch_size if batch_size > 1 else 2
    b = 2 * b / (batch_size - 1) - 1

    grid = torch.stack((x, y, b), dim=-1).view(1, 1, 1,-1, 3) # N x D_out x H_out x W_out x 3 (N=1, D_out=1, H_out=1)
    image_feat = image_feat.permute(1,0,2,3).unsqueeze(0) # N x C x D x H x W (N=1)

    image_feat_sampled = torch.nn.functional.grid_sample(image_feat,
                                                         grid=grid,
                                                         mode=image_sample_mode,
                                                         align_corners=True) # N x C x H_out x W_out (H_out=1, N=1)

    image_feat_sampled = image_feat_sampled.view(image_feat.shape[1], -1).t()

    return image_feat_sampled




