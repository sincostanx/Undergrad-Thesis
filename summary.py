import torch
import torch.nn as nn
import models
from option import args
from torchinfo import summary
import sys
import os
# import glob

# Ensemble for illustration only
# class Ensemble(nn.Module):
#     def __init__(self, args):
#         super(Ensemble, self).__init__()
#         self.adabins = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
#         self.bts = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
#         self.ldrn = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
#         self.controller = models.Controller.build(basemodel_name=args.encoder_controller, ensemble_size=len(args.baselearner_name), controller_input=args.controller_input, relax_linear_hypothesis=args.relax_linear_hypothesis)
#         self.controller_input = args.controller_input
    
#     def forward(self, x):
#         predList = [self.adabins(x.clone())[-1], self.bts(x.clone()), self.ldrn(x.clone())[-1]]
#         int_pred = torch.cat([nn.functional.interpolate(predList[0], x.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
#         if self.controller_input == "i": weight = self.controller(x.clone())
#         elif self.controller_input == "o": weight = self.controller(int_pred)
#         elif self.controller_input == "io": weight = self.controller(torch.cat([x.clone(), int_pred], dim=1))
#         pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(predList[1].shape)
#         return pred

# Generate dummy input and load network architecture
input_dim = 3
if args.controller_input == "o":
    input_dim = len(args.baselearner_name)
elif args.controller_input == "io":
    input_dim = 3 + len(args.baselearner_name)
input = torch.rand(1, input_dim, 416, 544)

if args.module == "adabins":
    model = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
elif args.module == "bts":
    model = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
elif args.module == "ldrn":
    model = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
elif args.module == "controller":
    model = models.Controller.build(basemodel_name=args.encoder_controller, ensemble_size=len(args.baselearner_name), controller_input=args.controller_input, relax_linear_hypothesis=args.relax_linear_hypothesis)
# elif args.module == "ensemble":
#     model = Ensemble(args)

basename = os.path.basename(args.save_dir)
save_path = os.path.join("./architecture", basename + ".txt")
os.makedirs("./architecture", exist_ok=True)

sys.stdout = open(save_path, "w")
summary(model, input_data=input, depth=4, col_names=("input_size", "output_size", "num_params"))
sys.stdout.close()

sys.stdout = sys.__stdout__
print("saved architecture at ", save_path)