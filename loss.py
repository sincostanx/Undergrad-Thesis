import numpy as np
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self, name=''):
        super(SILogLoss, self).__init__()
        self.name = 'SILog_' + name

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
    
class GradLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self, name=''):
        super(GradLoss, self).__init__()
        self.name = 'GradLoss_' + name

    @staticmethod
    def imgrad_loss(pred, gt, mask=None):
        N,C,_,_ = pred.size()
        grad_y, grad_x = GradLoss.imgrad(pred)
        grad_y_gt, grad_x_gt = GradLoss.imgrad(gt)
        grad_y_diff = torch.abs(grad_y - grad_y_gt)
        grad_x_diff = torch.abs(grad_x - grad_x_gt)
        if mask is not None:
            grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
            grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
        return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))

    @staticmethod
    def imgrad(img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

        if img.is_cuda:
            weight = weight.cuda()

        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x    
    
    def forward(self, input, target, mask=None):
        return GradLoss.imgrad_loss(input, target, mask)

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class TotalLoss:
    
    @staticmethod
    def getLoss(modules, args):
        """
        adabins --> [si, chamfer]
        bts, ldrn --> [si]
        controller --> [si, grad]
        """
        loss = [SILogLoss(name = args.module + "-SI-Loss")]
        if "adabins" in modules:
            loss.append(BinsChamferLoss())
        if "controller" in modules:
            loss.append(GradLoss(name = args.module + "-Grad-Loss"))
        return loss
    
    def __init__(self, modules, args):
        # modules = [args.module] + baselearners_name
        # check compatability between loss usage and weight param
        self.modules = modules
        self.w_chamfer = args.w_chamfer if args.chamfer else 0
        self.w_grad = 0.5 if args.gradient_loss else 0
        self.loss = getLoss(modules, args)
    
    def calculate(pred, depth, bin_edge, mask, device):
        result = []
        for loss in self.loss:
            if loss.name == self.modules[0] + "-SI-Loss":
                result.append(loss(pred, depth, mask=mask.to(torch.bool), interpolate=False))
            elif loss.name == self.modules[0] + "-Grad-Loss":
                if self.w_chamfer > 0: result.append(loss(pred, depth, mask))
                else: result.append(torch.Tensor([0]).to(device))
            elif loss.name == "ChamferLoss":
                result.append(loss((bin_edges, depth)))
        
        if self.modules[0] == "adabins": final_loss = result[0] + self.w_chamfer * result[1]
        elif self.module == "controller": final_loss = result[0] + self.w_grad * result[1]
        else: final_loss = result[0]
        
        return final_loss, result
        
