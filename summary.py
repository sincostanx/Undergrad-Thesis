import torch
import torch.nn as nn
import models
from option import args
from torchinfo import summary

# Ensemble for illustration only
class Ensemble(nn.Module):
    def __init__(self,n_bins, bts_size, min_val, max_val, norm, controller_input):
        super(Ensemble, self).__init__()
        self.adabins=models.UnetAdaptiveBins.build(basemodel_name="ghostnet_1x", n_bins=n_bins,min_val=min_val,max_val=max_val,norm=norm)
        self.bts=models.BtsModel.build(basemodel_name="ghostnet_1x", bts_size=bts_size, min_val=min_val, max_val=max_val,norm=norm)
        self.ldrn=models.LDRN.build(basemodel_name="ghostnet_1x", max_depth=max_val)
        self.controller=models.Controller.build(basemodel_name="ghostnet_1x", ensemble_size=3, controller_input=args.controller_input)
        self.controller_input = controller_input
    
    def forward(self, x):
        predList = [self.adabins(x.clone())[-1], self.bts(x.clone()), self.ldrn(x.clone())[-1]]
        int_pred = torch.cat([nn.functional.interpolate(predList[0], x.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
        if self.controller_input == "i": weight = self.controller(x.clone())
        elif self.controller_input == "o": weight = self.controller(int_pred)
        elif self.controller_input == "io": weight = self.controller(torch.cat([x.clone(), int_pred], dim=1))
        pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(predList[1].shape)
        return pred

# Generate dummy input and load network architecture
input = torch.rand(1, 3 + 3*(args.inspect_module == "controller" and args.controller_input == "io"), 416, 544)
if args.inspect_module == "adabins":
    model = models.UnetAdaptiveBins.build(basemodel_name="ghostnet_1x", n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
elif args.inspect_module == "bts":
    model = models.BtsModel.build(basemodel_name="ghostnet_1x", bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
elif args.inspect_module == "ldrn":
    model = models.LDRN.build(basemodel_name="ghostnet_1x", max_depth=args.max_depth)
elif args.inspect_module == "controller":
    model = models.Controller.build(basemodel_name="ghostnet_1x", ensemble_size=3, controller_input=args.controller_input)
elif args.inspect_module == "ensemble":
    model = Ensemble(n_bins=256, bts_size=512, min_val=0.001, max_val=10, norm="linear", controller_input=args.controller_input)

summary(model, input_data=input, depth=3, col_names=("input_size", "output_size", "num_params"))