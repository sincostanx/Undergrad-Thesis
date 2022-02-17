import glob
import os

from option import args
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

import model_io
import utils
import models
import random


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

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
    if args.module == "controller" and args.plot_weight: return (weight, pred)
    else: return pred

class InferenceHelper:
    def __init__(self, args, dataset='nyu', device='cuda:0'):
        self.toTensor = ToTensor()
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.saving_factor = 1000
        self.args = args
        
        args.gpu = int(args.gpu) if args.gpu is not None else 0
        args.distributed = False
        self.device = torch.device('cuda:{}'.format(args.gpu))
        
        self.baselearners = []
        self.baselearners_name = []
        if args.module == "adabins":
            self.model = models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
        elif args.module == "bts":
            self.model = models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)
        elif args.module == "ldrn":
            self.model = models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth)
        elif args.module == "controller":
            self.baselearners_name = args.baselearner_name
            self.baselearners_path = args.baselearner_path
            print("#"*40)
            print(self.baselearners_name)
            for x in self.baselearners_path:
                print(x)
            print("#"*40)
            
            for module in self.baselearners_name:
                if module == "adabins":
                    self.baselearners.append(models.UnetAdaptiveBins.build(basemodel_name=args.encoder, n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
                elif module == "bts":
                    self.baselearners.append(models.BtsModel.build(basemodel_name=args.encoder, bts_size=args.bts_size, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm))
                elif module == "ldrn":
                    self.baselearners.append(models.LDRN.build(basemodel_name=args.encoder, max_depth=args.max_depth))
            
            for model, path, module in zip(self.baselearners, self.baselearners_path, self.baselearners_name):
                model = model_io.load_pretrained(path, model, name=module, num_gpu=args.gpu)[0]
                model = model.cuda(args.gpu).eval()
            
            if not args.baseline:
                self.model=models.Controller.build(basemodel_name=args.encoder_controller, ensemble_size=len(self.baselearners_name), controller_input=args.controller_input, relax_linear_hypothesis=args.relax_linear_hypothesis)
            else:
                self.model = None
        
        if not args.baseline:
            self.model = model_io.load_pretrained(args.checkpoint_path, self.model, name=args.module, num_gpu=args.gpu)[0]
            self.model = self.model.cuda(args.gpu).eval()

    @torch.no_grad()
    def predict(self, img):
        
        temp = pred = predict(self.model, self.baselearners, self.baselearners_name, self.args, img)
        img = torch.Tensor(np.array(img.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred = predict(self.model, self.baselearners, self.baselearners_name, self.args, img, require_flip=True)
        
        if type(temp) == tuple:
            weight = 0.5*(pred[0]+temp[0]).squeeze().cpu().numpy()
            pred = 0.5*(pred[1]+temp[1])
            final = clipping(torch.Tensor(pred), self.args)
            return (weight, final)
        else:
            pred = 0.5*(pred+temp)
            final = clipping(torch.Tensor(pred), self.args)
            return final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir, get_stat=False):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True)
        if self.model != None: self.model.eval()
        
        plt.figure(figsize=(20,10))
        for f in tqdm(all_files):
            basename = os.path.basename(f).split('.')[0]
            
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)
            final = self.predict(image)
            
            if type(final) == tuple:
                weight = final[0]
                final = final[1]
                plt.clf()
                colors = ["darkred", "darkgreen", "darkblue", "darkorange", "crimson"]
                
                save_path = os.path.join(out_dir, basename + "-weight.npy")
                np.save(save_path, weight)
                
                for i in range(weight.shape[0]):
                    plt.subplot(1,weight.shape[0],i+1)
                    flat_img = weight[i].flatten()
                    plt.hist(flat_img, bins=100, range=(0,1), histtype="stepfilled", color=colors[i], weights=np.ones(flat_img.shape[0]) / flat_img.shape[0])
                
                save_path = os.path.join(out_dir, basename + "-weight.png")
                plt.savefig(save_path)
            
            save_path = os.path.join(out_dir, basename + ".png")
            
            plt.clf()
            plt.imshow(final, vmin=self.min_depth, vmax=self.max_depth)
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            
            # for saving predictions in the same format as ground truth depth maps
            # final = (final * self.saving_factor).astype('uint16')
            # Image.fromarray(final).save(save_path)
        
        if get_stat:
            ref_dir = args.ref_dir
            
            all_files = glob.glob(os.path.join(ref_dir, "**/*.jpg"), recursive=True)
            random.seed(args.seed)
            ref_num = len(all_files)
            all_files = random.sample(all_files, ref_num)
            
            start = time()
            for f in tqdm(all_files):
                image = np.asarray(Image.open(f), dtype='float32') / 255.
                image = transform(image).unsqueeze(0).to(self.device)
                final = self.predict(image)
            total_time = time() - start
            
            avg_time = total_time / len(all_files)
            fps = 1 / avg_time
        
            save_path = os.path.join(out_dir, "statistics.npy")
            np.save(save_path, np.array([avg_time, fps]))
            print(f"{out_dir} -----> Average time:{avg_time} seconds \t FPS : {fps}")


if __name__ == '__main__':
    inferHelper = InferenceHelper(args)
    pred = inferHelper.predict_dir(args.predict_dir, args.save_dir, get_stat=True)
    
