import os

import torch


def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        , fpath)


def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model

def load_pretrained(fpath, model, optimizer=None, name="none", num_gpu=-1):
    if num_gpu == -1: ckpt = torch.load(fpath, map_location="cpu")
    else: ckpt = torch.load(fpath, map_location="cuda:" + str(num_gpu))
    
    #print(ckpt.keys())
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():

        k_ = k
        if k_.startswith('module.'): k_ = k_.replace('module.', '')
        
        #k_ = name + "." + k_
        
        if name == "ldrn" and k_.startswith('decoder.'):
            idx = k_.rfind(".")
            if k_[idx-1].isnumeric(): k_ = k_[:idx-1] + 'module.' + k_[idx-1:]
        load_dict[k_] = v
    
    #print(ckpt.keys())
    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if name == "adabins" and k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif name == "adabins" and k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, epoch


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    print(ckpt.keys())
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        """if k.startswith('module.adabins.'):
            k_ = k.replace('module.adabins.', '')
            load_dict[k_] = v
        elif k.startswith('module.bts.'):
            k_ = k.replace('module.bts.', '')
            load_dict[k_] = v
        """
        #ldrn.decoder.ASPP.reduction1.module.0.weight
        #ldrn.decoder.ASPP.reduction1.0.weight
        
        k_ = k
        if k_.startswith('module.'): k_ = k_.replace('module.', '')
        if k_.startswith('ldrn.decoder.'):
            idx = k_.rfind(".")
            #print(idx)
            if k_[idx-1].isnumeric(): k_ = k_[:idx-1] + 'module.' + k_[idx-1:]
        load_dict[k_] = v
    
    #print(ckpt.keys())
    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adabins.adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adabins.adaptive_bins_layer.embedding_conv.',
                           'adabins.adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adabins.adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adabins.adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adabins.adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified, strict=False)
    return model, optimizer, epoch
