import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from option import args
import model_io
import models
import utils
from dataloader import DepthDataLoader
from utils import RunningAverageDict

def clipping(pred, args, require_flip=False):
    if require_flip: pred = np.array(pred.cpu().numpy())[..., ::-1].copy()
    else: pred = pred.squeeze().cpu().numpy()
    pred[pred < args.min_depth_eval] = args.min_depth_eval
    pred[pred > args.max_depth_eval] = args.max_depth_eval
    pred[np.isinf(pred)] = args.max_depth_eval
    pred[np.isnan(pred)] = args.min_depth_eval
    return pred

def predict(model, baselearners, baselearners_name, args, img, focal=torch.tensor([518.8579]), require_flip=False, require_clip=True):
    if args.module == "adabins":
        bin_edges, pred = model(img)
        pred = nn.functional.interpolate(pred, img.shape[-2:], mode="bilinear", align_corners=True)
    elif args.module == "bts":
        pred = model(img, focal)
    elif args.module == "ldrn":
        _, pred = model(img)
    elif args.module == "controller":                
        predList = []
        for i in range(len(baselearners)):
            # print(baselearners_name[i])
            if baselearners_name[i] == "adabins":
                bin_edges, pred = baselearners[i](img)
                predList.append(nn.functional.interpolate(pred, img.shape[-2:], mode="bilinear", align_corners=True))
            elif baselearners_name[i] == "bts": predList.append(baselearners[i](img, focal))
            elif baselearners_name[i] == "ldrn": predList.append(baselearners[i](img)[-1])
        
        int_pred = torch.cat(predList, dim = 1)
        # print(len(predList), int_pred.shape)
        if not args.baseline:
            if args.controller_input == "i": weight = model(img)
            elif args.controller_input == "o": weight = model(int_pred)
            elif args.controller_input == "io": weight = model(torch.cat([img, int_pred], dim=1))
            pred = torch.mul(int_pred, weight).sum(dim = 1).reshape((img.shape[0], 1, img.shape[2], img.shape[3]))
        else:
            pred = torch.mul(int_pred, 1./len(baselearners)).sum(dim = 1).reshape((img.shape[0], 1, img.shape[2], img.shape[3]))
    
    if require_clip: pred = clipping(pred,args,require_flip=require_flip)
    return pred

def validate(model, baselearners, baselearners_name, test_loader, args, gpus="cpu"):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]
    
    total_invalid = 0
    
    with torch.no_grad():
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader):
            img = batch["image"].to(device)
            focal = batch["focal"].to(device)
            depth = batch["depth"].to(device)
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    total_invalid += 1
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            
            temp = pred = predict(model, baselearners, baselearners_name, args, img)
            img = torch.Tensor(np.array(img.cpu().numpy())[..., ::-1].copy()).to(device)
            pred = predict(model, baselearners, baselearners_name, args, img, require_flip=True)
            
            #evaluate
            pred = 0.5*(pred+temp)
            final = clipping(torch.Tensor(pred),args)

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[
                        int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                        int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                    ] = 1

                elif args.eigen_crop:
                    if args.dataset == "kitti":
                        eval_mask[
                            int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                            int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                        ] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 4) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")
    
if __name__ == '__main__':
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data
    
    
    baselearners = []
    baselearners_name = []
    if args.module == "adabins":
        model = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    elif args.module == "bts":
        model = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
    elif args.module == "ldrn":
        model = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
    elif args.module == "controller":
        self.baselearners_name = args.baselearner_name
        self.baselearners_path = args.baselearner_path
        print("#"*40)
        print(self.baselearners_name)
        for x in self.baselearners_path:
            print(x)
        print("#"*40)

        for module in baselearners_name:
            if module == "adabins":
                baselearners.append(models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
            elif module == "bts":
                baselearners.append(models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
            elif module == "ldrn":
                baselearners.append(models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth))
        
        for model, path, module in zip(baselearners, baselearners_path, baselearners_name):
            model = model_io.load_pretrained(path, model, name=module, num_gpu=args.gpu)[0]
            model = model.cuda(args.gpu).eval()
        
        if not args.baseline:
            model=models.Controller.build(basemodel_name=args.encoder_controller, ensemble_size=len(baselearners_name), controller_input=args.controller_input, relax_linear_hypothesis=args.relax_linear_hypothesis)
        else:
            model = None
    
    if not args.baseline:
        model = model_io.load_pretrained(args.checkpoint_path, model, name=args.module, num_gpu=args.gpu)[0]
        model = model.cuda(args.gpu).eval()
    
    print("Evaluate {}".format(args.module))
    validate(model, baselearners, baselearners_name, test, args, gpus=[device])
