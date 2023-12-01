# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    require diffusers-0.11.1
"""
import os
import clip
import torch
from PIL import Image
from default_config import cfg as config
from models.lion_classifier import LION_Classifier
from utils.vis_helper import plot_points
import time
import datasets.pointflow_datasets as pf

model_path = './lion_ckpt/text2shape/chair/checkpoints/model.pt'
model_config = './lion_ckpt/text2shape/chair/cfg.yml'

config.merge_from_file(model_config)
lion = LION_Classifier(config)
lion.load_model(model_path)

def get_dataloader(cfg, ):
    loaders = pf.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    return train_loader, test_loader

if config.clipforge.enable:
    input_t = ["a swivel chair, five wheels"] 
    device_str = 'cuda'
    clip_model, clip_preprocess = clip.load(
                        config.clipforge.clip_model, device=device_str)    
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    print('clip_feat', clip_feat.shape)
else:
    clip_feat = None
t1 = time.time()
output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
t2 = time.time()
print("Execution time: ", t2-t1, "seconds")

# pts = output['points']
# img_name = "./tmp.png"
# plot_points(pts, output_name=img_name)
# img = Image.open(img_name)
# img.show()
