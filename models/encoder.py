import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def init_model(self, basemodel_name):
        self.basemodel_name = basemodel_name
        if self.basemodel_name == 'ghostnet_1x':
            self.basemodel = torch.hub.load("huawei-noah/ghostnet", "ghostnet_1x", pretrained=True)
            self.basemodel.global_pool = nn.Identity()
            self.basemodel.conv_head = nn.Identity()
            self.basemodel.act2 = nn.Identity()
            self.basemodel.classifier = nn.Identity()
        elif self.basemodel_name == 'mobilenet_v2':
            self.basemodel = torchvision.models.mobilenet_v2(pretrained=True)
            self.basemodel.classifier = nn.Identity()
            self.basemodel.avgpool = nn.Identity()
            self.basemodel.conv = nn.Identity()
            self.basemodel.features[-1] = nn.Identity()
        elif self.basemodel_name == 'densenet_161':
             self.basemodel = torchvision.models.densenet161(pretrained=True)
             # print(self.basemodel.classifier)
             self.basemodel.classifier = nn.Identity()
             # self.basemodel.norm5 = nn.Identity()
             # print(self.basemodel.classfier)
             # print(self.basemodel.features.conv0)
             # self.basemodel.avgpool = nn.Identity()
             # self.basemodel.conv = nn.Identity()
             # self.basemodel.features[-1] = nn.Identity()
    
    def modify_input(self, input_channel):
        if self.basemodel_name == 'ghostnet_1x':
            weight = self.basemodel.conv_stem.weight.clone()
            first_layer = nn.Conv2d(input_channel, 16, 3, 2, 1, bias=False)
        elif self.basemodel_name == 'mobilenet_v2':
            weight = self.basemodel.features[0][0].weight.clone()
            first_layer = nn.Conv2d(input_channel, 32, 3, 2, 1, bias=False)
        elif self.basemodel_name == 'densenet_161':
            weight = self.basemodel.features.conv0.weight.clone()
            first_layer = nn.Conv2d(input_channel, 96, 7, 2, 3, bias=False)
        
        with torch.no_grad():
            countx = input_channel//3
            for i in range(countx):
                first_layer.weight[:, 3*i:3*(i+1)] = weight
            if input_channel%3!=0:
                first_layer.weight[:, 3*countx:] = weight[: , 0:input_channel-3*countx]
    
        if self.basemodel_name == 'ghostnet_1x': self.basemodel.conv_stem = first_layer
        elif self.basemodel_name == 'mobilenet_v2': self.basemodel.features[0][0] = first_layer
        elif self.basemodel_name == 'densenet_161': self.basemodel.features.conv0 = first_layer
    
    def modify_output(self, output_channel):
        self.output_channel = output_channel
        if self.basemodel_name == 'ghostnet_1x':
            self.conv = nn.Conv2d(960, self.output_channel, kernel_size=1, stride=1, padding=0)
            self.feat_out_channels = [16, 24, 40, 112, self.output_channel]
            self.keep_feature = [-15,-13,-11,-9,-1] if self.output_channel != 960 else [-14,-12,-10,-8,-1]
        elif self.basemodel_name == 'mobilenet_v2':
            self.conv = nn.Conv2d(320, self.output_channel, kernel_size=1, stride=1, padding=0)
            self.feat_out_channels = [16, 24, 32, 64, self.output_channel]
            self.keep_feature = [2, 4, 7, 11, -1] if self.output_channel != 320 else [2, 4, 7, 11, -1]
        elif self.basemodel_name == 'densenet_161':
            self.conv = nn.Conv2d(2208, self.output_channel, kernel_size=1, stride=1, padding=0)
            self.feat_out_channels = [96, 96, 192, 384, self.output_channel]
            self.keep_feature = [-13, -11, -9, -7, -1] if self.output_channel != 2208 else [-13, -11, -9, -7, -1]
    
    def __init__(self, basemodel_name, output_channel, input_channel = 3):
        super(Encoder, self).__init__()
        self.init_model(basemodel_name)
        if input_channel !=3: self.modify_input(input_channel)
        self.modify_output(output_channel)

    def forward(self, x):
        features = [x]
        for k, v in self.basemodel._modules.items():
            # print(k,v)
            if self.basemodel_name == "ghostnet_1x" and k == "blocks":
                for ki, vi in v._modules.items(): features.append(vi(features[-1]))
            elif self.basemodel_name == "mobilenet_v2" and k == "features":
                for ki, vi in v._modules.items(): features.append(vi(features[-1]))
            elif self.basemodel_name == "densenet_161" and k == "features":
                for ki, vi in v._modules.items():
                    # print(ki,vi)
                    features.append(vi(features[-1]))
            else:
                # print("xxx",k,v)
                # print(features[-1].shape)
                features.append(v(features[-1]))
        if self.basemodel_name == 'ghostnet_1x' and self.output_channel!=960: features.append(self.conv(features[-1]))
        elif self.basemodel_name == 'mobilenet_v2' and self.output_channel!=320: features.append(self.conv(features[-1]))
        elif self.basemodel_name == 'densenet_161' and self.output_channel!=2208: features.append(self.conv(features[-1]))
        # for x in features: print(x.shape)
        return [features[x] for x in self.keep_feature]
