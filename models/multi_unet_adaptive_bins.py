import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT


def depthwise(in_channels, kernel_size, activation="relu"):
    padding = (kernel_size - 1) // 2
    assert (
        2 * padding == kernel_size - 1
    ), "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    layer = nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            groups=in_channels,
        ),
        nn.BatchNorm2d(in_channels),
    )
    if activation=="relu": layer.add_module("relu", nn.ReLU(inplace=True))
    elif activation=="elu": layer.add_module("elu", nn.ELU(inplace=True))
    elif activation=="sigmoid": layer.add_module("sigmoid", nn.Sigmoid())
    elif activation=="leaky_relu": layer.add_module("leaky_relu", nn.LeakyReLU())
    return layer


def pointwise(in_channels, out_channels, activation="relu"):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
    )
    if activation=="relu": layer.add_module("relu", nn.ReLU(inplace=True))
    elif activation=="elu": layer.add_module("elu", nn.ELU(inplace=True))
    elif activation=="sigmoid": layer.add_module("elu", nn.Sigmoid())
    elif activation=="leaky_relu": layer.add_module("leaky_relu", nn.LeakyReLU())
    return layer


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(
            depthwise(skip_input, 3,"leaky_relu"), pointwise(skip_input, output_features,"leaky_relu")
        )

    def forward(self, x, concat_with):
        x = self._net(x)
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        return torch.cat([up_x, concat_with], dim=1)
        """
        up_x = F.interpolate(x, size=[concat_with.size(
            2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)
        """


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        """
        self.up1 = UpSampleBN(skip_input=features // 1 +
                              112, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 +
                              40, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 +
                              24, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 +
                              16, output_features=features // 16)
        """

        self.up1 = UpSampleBN(skip_input=features // 1, output_features=features // 2)
        self.up2 = UpSampleBN(
            skip_input=features // 2 + 112, output_features=features // 4
        )
        self.up3 = UpSampleBN(
            skip_input=features // 4 + 40, output_features=features // 8
        )
        self.up4 = UpSampleBN(
            skip_input=features // 8 + 24, output_features=features // 16
        )

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)

        self.conv3 = nn.Sequential(
            depthwise(features // 16 + 16, 3), pointwise(features // 16 + 16, num_classes,"none")
        )
        """
        self.conv3 = nn.Conv2d(
            features // 16 + 16, num_classes, kernel_size=3, stride=1, padding=1
        )
        """

        """
        self.conv3 = nn.Conv2d(features // 16, num_classes,
                               kernel_size=3, stride=1, padding=1)
        """

        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        #for x in features: print(x.size())
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[-15],
            features[-13],
            features[-11],
            features[-9],
            features[-1],
        )
        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


"""
class Decoder(nn.Module):
    def __init__(self, num_features=960, num_classes=1, bottleneck_features=960):

        super(Decoder, self).__init__()
        features = int(num_features)
        kernel_size = 5
        self.decode_conv1 = nn.Sequential(
            depthwise(features // 1, kernel_size),
            pointwise(features // 1, features // 2),
        )
        self.decode_conv2 = nn.Sequential(
            depthwise(features // 2, kernel_size),
            pointwise(features // 2, features // 4),
        )
        self.decode_conv3 = nn.Sequential(
            depthwise(features // 4, kernel_size),
            pointwise(features // 4, features // 8),
        )
        self.decode_conv4 = nn.Sequential(
            depthwise(features // 8, kernel_size),
            pointwise(features // 8, features // 16),
        )
        self.decode_conv5 = nn.Sequential(
            depthwise(features // 16, kernel_size),
            pointwise(features // 16, features // 32),
        )
        self.conv3 = nn.Conv2d(
            features // 32, num_classes, kernel_size=3, stride=1, padding=1
        )

    def forward(self, features):
        x = features[-1]
        for i in range(1, 6):
            layer = getattr(self, "decode_conv{}".format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        x = self.conv3(x)
        return x
"""


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend
        self.conv = nn.Conv2d(960, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        features.append(self.conv(features[-1]))
        return features


class MultiUnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm="linear"):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder1 = Encoder(backend)
        self.encoder2 = Encoder(backend)
        self.encoder3 = Encoder(backend)
        self.encoder4 = Encoder(backend)
        self.adaptive_bins_layer = mViT(
            128,
            n_query_channels=128,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128,
            norm=norm,
        )

        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

    def forward(self, x, **kwargs):
        #self.encoder(x)

        unet_out = self.decoder(torch.cat([self.encoder1(x.clone()), self.encoder2(x.clone()), self.encoder3(x.clone()), self.encoder4(x.clone())], dim=1), **kwargs)
        # print(unet_out.size())
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (
            self.max_val - self.min_val
        ) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_val
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        """
        print('Loading GhostNet model...')
        basemodel = ghostnet()
        print('Done.')
        """
        basemodel_name = "ghostnet_1x"
        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "huawei-noah/ghostnet", "ghostnet_1x", pretrained=True
        )
        print("Done.")
        #print(basemodel)

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.conv_head = nn.Identity()
        basemodel.act2 = nn.Identity()
        basemodel.classifier = nn.Identity()
        #print(basemodel)
        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print("Done.")
        return m


if __name__ == "__main__":

    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
