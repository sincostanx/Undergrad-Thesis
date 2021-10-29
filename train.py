import os
import uuid
from datetime import datetime as dt

import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

from option import args
import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage

PROJECT = "Thesis_MixEnsemble"
logging = True

def is_rank_zero(args):
    return args.rank == 0

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    baselearners = []
    if args.module == "adabins":
        model = models.UnetAdaptiveBins.build(basemodel_name="ghostnet_1x", n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    elif args.module == "bts":
        model = models.BtsModel.build(basemodel_name="ghostnet_1x", bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
    elif args.module == "ldrn":
        model = models.LDRN.build(basemodel_name="ghostnet_1x", max_depth=args.max_depth)
    elif args.module == "controller":
        baselearners_name = ["adabins", "bts", "ldrn"]
        baselearners_path = [
            "./checkpoints/UnetAdaptiveBins_04-Aug_21-32-nodebs32-tep25-lr0.000613590727341-wd0.0001-maxMT0.95-seed20210804-adabins-c424a732-f55f-47ef-9d1d-02e2eaa7af05_best.pt",
            "./checkpoints/UnetAdaptiveBins_05-Aug_06-49-nodebs32-tep25-lr0.000863310742922-wd0.0001-maxMT0.95-seed0-bts-b976954d-f4ef-48f7-8468-848a59d08692_best.pt",
            "./checkpoints/UnetAdaptiveBins_05-Aug_17-50-nodebs32-tep25-lr0.001232846739442-wd0.0001-maxMT0.95-seed1234567890-ldrn-1d367e30-91c5-4b3a-b06c-733221a79e7e_best.pt"
        ]
        baselearners.append(models.UnetAdaptiveBins.build(basemodel_name="ghostnet_1x", n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
        baselearners.append(models.BtsModel.build(basemodel_name="ghostnet_1x", bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
        baselearners.append(models.LDRN.build(basemodel_name="ghostnet_1x", max_depth=args.max_depth))
        for i in range(len(baselearners)):
            baselearners[i] = model_io.load_pretrained(baselearners_path[i], baselearners[i], name=baselearners_name[i], num_gpu=args.gpu)[0]
        
        model=models.Controller.build(basemodel_name="ghostnet_1x", ensemble_size=3, controller_input=args.controller_input)
    
    ###################################### Distributed Training ##############################################
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        for i in range(len(baselearners)):
            baselearners[i] = baselearners[i].cuda(args.gpu)
        
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
        
        for i in range(len(baselearners)):
            baselearners[i] = nn.SyncBatchNorm.convert_sync_batchnorm(baselearners[i])
            baselearners[i] = baselearners[i].cuda(args.gpu)
            baselearners[i] = torch.nn.parallel.DistributedDataParallel(baselearners[i], device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        for i in range(len(baselearners)):
            baselearners[i] = baselearners[i].cuda()
            baselearners[i] = torch.nn.DataParallel(baselearners[i])
        
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    experiment_name = args.module + args.controller_input if args.module == "controller" else args.module
    train(model, baselearners, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root, experiment_name=experiment_name, optimizer_state_dict=None)


def train(model, baselearners, args, epochs=10, experiment_name="controller", lr=0.0001, root=".", device=None, optimizer_state_dict=None):
    print("Training {}, use {} base learners".format(args.module, len(baselearners)))
    
    global PROJECT
    if device is None:
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-maxMT{args.max_momentum}-seed{args.seed}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = (not args.distributed) or args.rank == 0
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(",") if args.tags != "" else None
        if args.dataset != "nyu":
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(
            project=PROJECT,
            name=name,
            config=args,
            dir=args.root,
            tags=tags,
            notes=args.notes,
            settings=wandb.Settings(start_method="fork"),
        )
        # wandb.watch(model)
    
    ###################################### Data loader ##############################################
    train_loader = DepthDataLoader(args, "train").data
    test_loader = DepthDataLoader(args, "online_eval").data

    ###################################### Loss Functions ##############################################
    criterion_si = SILogLoss(name = args.module + "-SI-Loss")
    if args.module == "adabins":
        criterion_bins = BinsChamferLoss() if args.chamfer else None
    
    ###################################### Freeze base learners ##############################################
    model.train()
    for m in baselearners: m.eval()
    
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
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    steps_per_epoch = len(train_loader)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=args.max_momentum,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )
    if args.resume != "" and scheduler is not None:
        scheduler.step(args.epoch + 1)
    
    ################################# Train loop ##########################################################
    for epoch in range(args.epoch, epochs):
        #for reproducibility
        train_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)    
        torch.manual_seed(epoch)  
        torch.cuda.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)
        
        if should_log:
            wandb.log({"Epoch": epoch}, step=step)
        for i, batch in (
            tqdm(
                enumerate(train_loader),
                desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                total=len(train_loader),
            )
            if is_rank_zero(args)
            else enumerate(train_loader)
        ):

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
            elif args.module == "controller":
                predList = [baselearners[0](img)[-1], baselearners[1](img, focal), baselearners[2](img)[-1]]
                int_pred = torch.cat([nn.functional.interpolate(predList[0], depth.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
                if args.controller_input == "i": weight = model(img)
                elif args.controller_input == "o": weight = model(int_pred)
                elif args.controller_input == "io": weight = model(torch.cat([img, int_pred], dim=1))
                pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(depth.size())
            
            mask = depth > args.min_depth
            
            require_interpolate = True if args.module == "adabins" else False
            loss_si = criterion_si(pred, depth, mask=mask.to(torch.bool), interpolate=require_interpolate)
            if args.module == "adabins":
                loss_bins = criterion_bins(bin_edges, depth) if args.w_chamfer > 0 else torch.Tensor([0]).to(img.device)
            
            loss = loss_si + args.w_chamfer * loss_bins if args.module == "adabins" else loss_si
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_si.name}": loss_si.item()}, step=step)
                #wandb.log({f"Train/{silog_criterion.name}": l_silog.item()}, step=step)

            step += 1
            scheduler.step()

            ########################################################################################################
            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_total_si= validate(args, model, baselearners, test_loader, criterion_si, epoch, epochs, device)
                if should_log:
                    wandb.log(
                        {
                            f"Test/{criterion_si.name}": val_total_si.get_value(),
                            # f"Test/{criterion_adabins_bins.name}": val_bins.get_value()
                        },
                        step=step,
                    )

                    wandb.log(
                        {f"Metrics/{k}": v for k, v in metrics.items()}, step=step
                    )
                    model_io.save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        f"{experiment_name}_{run_id}_latest.pt",
                        root=os.path.join(root, "checkpoints"),
                    )

                if metrics["abs_rel"] < best_loss and should_write:
                    model_io.save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        f"{experiment_name}_{run_id}_best.pt",
                        root=os.path.join(root, "checkpoints"),
                    )
                    best_loss = metrics["abs_rel"]
                model.train()

    wandb.finish()
    return model

     
def validate(args, model, baselearners, test_loader, criterion_si, epoch, epochs, device="cpu"):
    with torch.no_grad():
        val_total_si = RunningAverage()
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
            elif args.module == "controller":
                predList = [baselearners[0](img)[-1], baselearners[1](img, focal), baselearners[2](img)[-1]]
                int_pred = torch.cat([nn.functional.interpolate(predList[0], depth.shape[-2:], mode="bilinear", align_corners=True), predList[1], predList[2]], dim = 1)
                if args.controller_input == "i": weight = model(img)
                elif args.controller_input == "o": weight = model(int_pred)
                elif args.controller_input == "io": weight = model(torch.cat([img, int_pred], dim=1))
                pred = torch.mul(int_pred, weight).sum(dim = 1).reshape(depth.size())
            
            mask = depth > args.min_depth
                
            loss_si = criterion_si(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            val_total_si.append(loss_si.item())
            
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

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
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_total_si

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
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
