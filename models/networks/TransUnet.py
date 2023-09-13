import torch
# from models.unetresnet import UNetResNet
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg



def get_transNet(n_classes):
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    # net.load_from(weights=np.load(config_vit.pretrained_path))
   # initialize_weights(net)
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    # ckpt = np.load(config_vit.pretrained_path)
    # new_model_static = net.state_dict()
    # new_model_static.update(ckpt)
    # net.load_state_dict(new_model_static)
    #print('*'*100)
    #print('self.net.static:', net.state_dict())
    #print(config_vit.pretrained_path)
    return net


if __name__ == '__main__':
    net = get_transNet(12)
    img = torch.randn((2, 3, 512, 512))
    segments = net(img)
    print(segments.size())
