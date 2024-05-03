import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from PIL import Image

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LSeg(BaseModel):
    def __init__(
        self,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv_1 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_2 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_3 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_4 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.text = clip.tokenize(self.labels)    
        
    def forward(self, x, labelset=''):
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)    
        
        # print(f"Text (after tokenize) length: {len(text)}") # 4
        # print(f"Image shape: {x.shape}") # [1, 3, 416, 416] # 416x416 is the input size

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        # Encode text features
        text_features = self.clip_pretrained.encode_text(text)
        # print(f"Text features shape: {text_features.shape}") # [4, 512] # 4 is the number of token in a label

        # Get image features
        image_features = self.scratch.head1(path_1)
        # print(f"Image features shape: {image_features.shape}") # [1, 512, 208, 208] # 208x208 is the W/2xH/2 size of the input
        
        # Visualize image features
        fig, ax = plt.subplots(nrows=2, ncols=5)
        for r, row in enumerate(ax):
            for c, col in enumerate(row):
                img = image_features[0][r+c].detach().cpu().numpy()
                # print(f"Channel #{r*len(row)+c}: max: {img.max()}, min: {img.min()}")
                # Normalize image
                img = (img - img.min()) / (img.max() - img.min())
                col.imshow(img, cmap='gray')

        plt.savefig("image_features.png")

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)
        # print(f"Image features shape (after reshaped and permute): {image_features.shape}") # [43264, 512] 

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # print(f"Logit scale shape: {self.logit_scale.shape}") # []

        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()
        # print(f"Logits per image shape: {logits_per_image.shape}") # [43264, 4]

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2) 

        img = out[0][0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        plt.savefig("logits_per_image.png")

        # print(f"Out (before headblock) shape: {out.shape}") # [1, 4, 208, 208]

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        # print(f"Out (after headblock) shape: {out.shape}") # [1, 4, 208, 208]

        out_1 = self.scratch.output_conv_1(out)
        out_2 = self.scratch.output_conv_2(out)
        out_3 = self.scratch.output_conv_3(out)
        out_4 = self.scratch.output_conv_4(out)
        
        final_out = out_1 + out_2 + out_3 + out_4

        return final_out


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        super().__init__(**kwargs)

        if path is not None:
            self.load(path)


    
        
    