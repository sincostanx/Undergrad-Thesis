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

def validate(model, baselearners, test_loader, args, gpus="cpu"):
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
            
            # original image
            if args.module == "adabins":
                bin_edges, pred = model(img)
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            elif args.module == "bts":
                pred = model(img, focal)
            elif args.module == "ldrn":
                _, pred = model(img)
            elif args.module == "controller":
                predList = [baselearners[0](img)[-1], baselearners[1](img, focal), baselearners[2](img)[-1]]
                int_pred = torch.cat([nn.functional.interpolate(predList[0], depth.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
                if not args.baseline:
                    if args.controller_input == "i": weight = model(img)
                    elif args.controller_input == "o": weight = model(int_pred)
                    elif args.controller_input == "io": weight = model(torch.cat([img, int_pred], dim=1))
                    pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(depth.size())
                else:
                    pred = torch.mul(int_pred, 1/3.).sum(dim = 1).reshape(depth.size())
                
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval
            temp = pred
            
            #flipped image
            """
            These lines of code function properly, however they are commented out to be consistent with the progress described in the report.
            In the complete thesis, models' prediction will based on both original image and its flipped version.
            """
            # img = torch.Tensor(np.array(img.cpu().numpy())[..., ::-1].copy()).to(device)
            
            # if args.module == "adabins":
            #     bin_edges, pred = model(img)
            #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            # elif args.module == "bts":
            #     pred = model(img, focal)
            # elif args.module == "ldrn":
            #     _, pred = model(img)
            # elif args.module == "controller":
            #     predList = [baselearners[0](img)[-1], baselearners[1](img, focal), baselearners[2](img)[-1]]
            #     int_pred = torch.cat([nn.functional.interpolate(predList[0], depth.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
            #     if not args.baseline:
            #         if args.controller_input == "i": weight = model(img)
            #         elif args.controller_input == "o": weight = model(int_pred)
            #         elif args.controller_input == "io": weight = model(torch.cat([img, int_pred], dim=1))
            #         pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(depth.size())
            #     else:
            #         pred = torch.mul(int_pred, 1/3.).sum(dim = 1).reshape(depth.size())
            
            # pred = np.array(pred.cpu().numpy())[..., ::-1].copy()
            # pred[pred < args.min_depth_eval] = args.min_depth_eval
            # pred[pred > args.max_depth_eval] = args.max_depth_eval
            # pred[np.isinf(pred)] = args.max_depth_eval
            # pred[np.isnan(pred)] = args.min_depth_eval
            
            #evaluate
            pred = 0.5*(pred+temp)
            
            final = torch.Tensor(pred)
            final = final.squeeze().cpu().numpy()
            final[final < args.min_depth_eval] = args.min_depth_eval
            final[final > args.max_depth_eval] = args.max_depth_eval
            final[np.isinf(final)] = args.max_depth_eval
            final[np.isnan(final)] = args.min_depth_eval

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
    if args.module == "adabins":
        model = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    elif args.module == "bts":
        model = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
    elif args.module == "ldrn":
        model = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
    elif args.module == "controller":
        baselearners_name = ["adabins", "bts", "ldrn"]
        baselearners_path = [
            "./checkpoints/UnetAdaptiveBins_04-Aug_21-32-nodebs32-tep25-lr0.000613590727341-wd0.0001-maxMT0.95-seed20210804-adabins-c424a732-f55f-47ef-9d1d-02e2eaa7af05_best.pt",
            "./checkpoints/UnetAdaptiveBins_05-Aug_06-49-nodebs32-tep25-lr0.000863310742922-wd0.0001-maxMT0.95-seed0-bts-b976954d-f4ef-48f7-8468-848a59d08692_best.pt",
            "./checkpoints/UnetAdaptiveBins_05-Aug_17-50-nodebs32-tep25-lr0.001232846739442-wd0.0001-maxMT0.95-seed1234567890-ldrn-1d367e30-91c5-4b3a-b06c-733221a79e7e_best.pt"
        ]
        baselearners.append(models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
        baselearners.append(models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
        baselearners.append(models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth))
        for i in range(len(baselearners)):
            baselearners[i] = model_io.load_pretrained(baselearners_path[i], baselearners[i], name=baselearners_name[i], num_gpu=args.gpu)[0]
            baselearners[i] = baselearners[i].cuda(args.gpu).eval()
        
        model=models.Controller.build(basemodel_name=args.encoder_controller, ensemble_size=3, controller_input=args.controller_input)
    
    model = model_io.load_pretrained(args.checkpoint_path, model, name=args.module, num_gpu=args.gpu)[0]
    model = model.cuda(args.gpu).eval()
    
    print("Evaluate {}".format(args.module))
    validate(model, baselearners, test, args, gpus=[device])
