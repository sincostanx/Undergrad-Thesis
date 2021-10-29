import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT
from .layers import SeperableConv2d
from .encoder import Encoder

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = SeperableConv2d(skip_input, output_features, kernel_size = 3, activation = ["leaky_relu", "leaky_relu"])

    def forward(self, x, concat_with):
        """
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)
        """
        
        x = self._net(x)
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode="bilinear", align_corners=True)
        return torch.cat([up_x, concat_with], dim=1)
        

class DecoderBN(nn.Module):
    def __init__(self, feat_out_channels, num_features=2048, num_classes=1):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(feat_out_channels[4], features, kernel_size=1, stride=1, padding=1)
        
        """
        self.up1 = UpSampleBN(skip_input=features // 1 + feat_out_channels[3], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + feat_out_channels[2], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + feat_out_channels[1], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + feat_out_channels[0], output_features=features // 16)
        
        self.conv3 = SeperableConv2d(features // 16, num_classes, kernel_size = 3, activation = ["relu", "none"])
        """
        
        
        self.up1 = UpSampleBN(skip_input=features // 1, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 112, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 40, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 24, output_features=features // 16)
        
        self.conv3 = SeperableConv2d(features // 16 + 16, num_classes, kernel_size = 3, activation = ["relu", "none"])
        

    def forward(self, features):
        x_d0 = self.conv2(features[4])
        x_d1 = self.up1(x_d0, features[3])
        x_d2 = self.up2(x_d1, features[2])
        x_d3 = self.up3(x_d2, features[1])
        x_d4 = self.up4(x_d3, features[0])
        out = self.conv3(x_d4)
        return out

class UnetAdaptiveBins(nn.Module):
    def __init__(self, basemodel_name, n_bins=100, min_val=0.1, max_val=10, norm="linear"):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(basemodel_name, output_channel=2048)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16, dim_out=n_bins, embedding_dim=128, norm=norm)
        self.decoder = DecoderBN(feat_out_channels=self.encoder.feat_out_channels, num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x, **kwargs):

        unet_out = self.decoder(self.encoder(x), **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode="constant", value=self.min_val)
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
    def build(cls, basemodel_name, n_bins, **kwargs):
        m = cls(basemodel_name, n_bins=n_bins, **kwargs)
        return m

