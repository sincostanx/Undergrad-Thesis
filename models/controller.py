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
        x = self._net(x)
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode="bilinear", align_corners=True)
        return torch.cat([up_x, concat_with], dim=1)

class DecoderBN(nn.Module):
    def __init__(self, feat_out_channels, ensemble_size=3, num_features=960, relax_linear_hypothesis=False):
        super(DecoderBN, self).__init__()
        self.relax_linear_hypothesis = relax_linear_hypothesis
        
        features = int(num_features)

        self.conv2 = nn.Conv2d(feat_out_channels[-1], features, kernel_size=1, stride=1, padding=1)
        
        self.up1 = UpSampleBN(skip_input=features // 1, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + feat_out_channels[-2], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + feat_out_channels[-3], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + feat_out_channels[-4], output_features=features // 16)
        self.up5 = SeperableConv2d(features // 16 + feat_out_channels[-5], features // 32, kernel_size = 3, activation = ["leaky_relu", "leaky_relu"])
        
        self.conv3 = SeperableConv2d(features // 32, ensemble_size, kernel_size = 3, activation = ["relu", "none"])
        
        if relax_linear_hypothesis:
            print("Relax linear hypothesis!")
            self.act_out1 = nn.ReLU(inplace=True)
            self.conv_out = nn.Conv2d(ensemble_size, 1, kernel_size=1, stride=1)
            self.act_out2 = nn.ReLU(inplace=True)
        else: self.softmax = nn.Softmax2d()

    def forward(self, features):
        x_d0 = self.conv2(features[4])
        x_d1 = self.up1(x_d0, features[3])
        x_d2 = self.up2(x_d1, features[2])
        x_d3 = self.up3(x_d2, features[1])
        x_d4 = self.up4(x_d3, features[0])
        x_d5 = self.up5(x_d4)
        x_d5 = F.interpolate(x_d5, size=[2*features[0].size()[2],2*features[0].size()[3]],mode="bilinear", align_corners=True)
        
        # numerical stable softmax for weight mask
        out = self.conv3(x_d5)
        if self.relax_linear_hypothesis:
            # print(out)
            out = self.act_out2(self.conv_out(self.act_out1(out)))
            out = out + torch.full_like(out,1e-9)
            # print(out)
        else:
            maxx = torch.max(out, dim = 1)[0].resize(out.size()[0],1,out.size()[2],out.size()[3])
            out = self.softmax(torch.subtract(out,maxx))
        return out


class Controller(nn.Module):
    def __init__(self, basemodel_name, ensemble_size=3, controller_input="io", relax_linear_hypothesis=False):
        super(Controller, self).__init__()
        self.ensemble_size=ensemble_size
        if controller_input == "i":
            self.encoder = Encoder(basemodel_name, output_channel=960, input_channel=3)
        elif controller_input == "o":
            self.encoder = Encoder(basemodel_name, output_channel=960, input_channel=ensemble_size)
        elif controller_input == "io":
            self.encoder = Encoder(basemodel_name, output_channel=960, input_channel=3+ensemble_size)
        self.decoder = DecoderBN(feat_out_channels=self.encoder.feat_out_channels, ensemble_size=ensemble_size, relax_linear_hypothesis=relax_linear_hypothesis)

    def forward(self, x, **kwargs):
        # return self.encoder(x)
        return self.decoder(self.encoder(x), **kwargs)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, basemodel_name, ensemble_size, controller_input, relax_linear_hypothesis, **kwargs):
        m = cls(basemodel_name, ensemble_size=ensemble_size, controller_input=controller_input, relax_linear_hypothesis=relax_linear_hypothesis, **kwargs)
        return m

