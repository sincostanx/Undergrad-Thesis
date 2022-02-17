# This code has not been refactored yet. So, it may not run properly...
import os

import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import _LRScheduler
import csv
from tqdm import tqdm

from option import args
import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

def is_rank_zero(args):
    return args.rank == 0

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    if args.module == "adabins":
        model = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    elif args.module == "bts":
        model = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
    elif args.module == "ldrn":
        model = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
    
    ###################################### Distributed Training ##############################################
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None, optimizer_state_dict=None):
    if device is None:
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
  
    ###################################### Data loader ##############################################
    train_loader = DepthDataLoader(args, "train").data
    test_loader = DepthDataLoader(args, "online_eval").data
    
    ###################################### Loss Functions ##############################################
    criterion_si = SILogLoss(name = args.module + "-SI-Loss")
    if args.module == "adabins":
        criterion_bins = BinsChamferLoss() if args.chamfer else None
    
    model.train()
    
    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [
            {"params": m.get_1x_lr_params(), "lr": lr / 10},
            {"params": m.get_10x_lr_params(), "lr": lr},
        ]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    ###################################### Some globals ###############################################
    iters = len(train_loader)
    step = args.epoch * iters
    final_metrics = []

    ###################################### Scheduler ###############################################
    LIMIT = args.lrtest
    scheduler = ExponentialLR(optimizer, 5e-3, LIMIT)
    
    ################################# Train loop ##########################################################
    for epoch in range(args.epoch, epochs):
        #for reproducibility
        train_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)    
        torch.manual_seed(epoch)  
        torch.cuda.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)
        
        for i, batch in (
            tqdm(
                enumerate(train_loader),
                desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                total=len(train_loader),
            )
            if is_rank_zero(args)
            else enumerate(train_loader)
        ):

            if step >= LIMIT:
                fieldnames = [k for k, v in final_metrics[0].items()]            
                with open(args.module + '-lr-test-result.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(final_metrics)
                return model
            
            optimizer.zero_grad()
            img = batch["image"].to(device)
            focal = batch["focal"].to(device)
            depth = batch["depth"].to(device)
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    continue

            if args.module == "adabins":
                bin_edges, pred = model(img)
            elif args.module == "bts":
                pred = model(img, focal)
            elif args.module == "ldrn":
                _, pred = model(img)
            
            mask = depth > args.min_depth
            
            require_interpolate = True if args.module == "adabins" else False
            loss_si = criterion_si(pred, depth, mask=mask.to(torch.bool), interpolate=require_interpolate)
            if args.module == "adabins":
                loss_bins = criterion_bins(bin_edges, depth) if args.w_chamfer > 0 else torch.Tensor([0]).to(img.device)
            
            loss = loss_si + args.w_chamfer * loss_bins if args.module == "adabins" else loss_si
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            
            model.eval()
            metrics, val_si = validate(args, model, test_loader, criterion_si, epoch, epochs, device)
            metrics["learning_rate_encoder"] = scheduler.get_last_lr()[0]
            metrics["learning_rate_actual"] = scheduler.get_last_lr()[1]
            metrics[args.module + "_loss"] = val_si.get_value()
            final_metrics.append(metrics)

            step += 1
            scheduler.step()
            model.train()
            
    return model


def validate(args, model, test_loader, criterion_si, epoch, epochs, device="cpu"):
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in (
            tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation")
            if is_rank_zero(args)
            else test_loader
        ):
            img = batch["image"].to(device)
            focal = batch["focal"].to(device)
            depth = batch["depth"].to(device)
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            
            if args.module == "adabins":
                bin_edges, pred = model(img)
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            elif args.module == "bts":
                pred = model(img, focal)
            elif args.module == "ldrn":
                _, pred = model(img)
                
            mask = depth > args.min_depth

            loss_si = criterion_si(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            val_si.append(loss_si.item())
            
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval
            )
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
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


if __name__ == "__main__":
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = "train"
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")

        args.world_size = len(nodes)
        args.rank = int(os.environ["SLURM_PROCID"])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method("forkserver")

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = "tcp://{}:{}".format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = "nccl"
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1: args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
