import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, basemodel_name, output_channel, input_channel = 3):
        super(Encoder, self).__init__()
        if basemodel_name == 'ghostnet_1x':
            self.basemodel = torch.hub.load("huawei-noah/ghostnet", "ghostnet_1x", pretrained=True)
            self.basemodel.global_pool = nn.Identity()
            self.basemodel.conv_head = nn.Identity()
            self.basemodel.act2 = nn.Identity()
            self.basemodel.classifier = nn.Identity()
            
            if input_channel !=3 :
                weight = self.basemodel.conv_stem.weight.clone()
                self.basemodel.conv_stem = nn.Conv2d(input_channel, 16, 3, 2, 1, bias=False)
                with torch.no_grad():
                    countx = input_channel//3
                    for i in range(countx):
                        self.basemodel.conv_stem.weight[:, 3*i:3*(i+1)] = weight
                    if input_channel%3!=0:
                        self.basemodel.conv_stem.weight[:, 3*countx:] = weight[: , 0:input_channel-3*countx]
            
        self.output_channel = output_channel
        self.conv = nn.Conv2d(960, self.output_channel, kernel_size=1, stride=1, padding=0)
        self.feat_out_channels = [16, 24, 40, 112, self.output_channel]
        if self.output_channel != 960: self.keep_feature = [-15,-13,-11,-9,-1]
        else: self.keep_feature = [-14,-12,-10,-8,-1]

    def forward(self, x):
        features = [x]
        for k, v in self.basemodel._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else: features.append(v(features[-1]))
        if self.output_channel!=960: features.append(self.conv(features[-1]))
        return [features[x] for x in self.keep_feature]
