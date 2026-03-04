import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora_lib
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0, merge_weights=False, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
            
        self.lora_A = nn.Parameter(torch.Tensor(in_features, r))
        self.lora_B = nn.Parameter(torch.Tensor(r, out_features))
        self.scaling = lora_alpha / r
        
        self.reset_parameters()
        
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        if self.r > 0:
            nn.init.normal_(self.lora_A, mean=0, std=0.02)
            nn.init.zeros_(self.lora_B)    
    
    def forward(self, x):
        original = F.linear(x, self.weight, self.bias)
        
        if self.r > 0 and not self.merge_weights:
            lora_x = self.lora_dropout(x)
            lora_output = (lora_x @ self.lora_A @ self.lora_B) * self.scaling
            return original + lora_output
        
        return original
    
    def train(self, mode=True):
        return super(Linear, self).train(mode) 
            
class LoRAModule:
    Linear = Linear
    
lora = LoRAModule()
        
        
@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.use_loralib = config.get('use_loralib', False) if config else False
        
        self.backbone = self.build_backbone(config)
        
        # 根据配置选择使用 loralib 还是自编 LoRA
        LinearClass = lora_lib.Linear if self.use_loralib else lora.Linear
        
        self.head = LinearClass(
            in_features=1024,
            out_features=2,
            r=2,
            lora_alpha=8,
            lora_dropout=0,
            merge_weights=False,
            bias=True
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # ⚠⚠⚠ Download CLIP model using the below link
        # https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link 
        
        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        
        # ViT-L/14 224*224
        clip_model = CLIPModel.from_pretrained("/home/user1/effort/effort_main/Effort-AIGI-Detection-main/DeepfakeBench/training/models--openai--clip-vit-large-patch14")

        for param in clip_model.vision_model.parameters():
            param.requires_grad = False

        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        
        # 根据配置选择使用 loralib 还是自编 LoRA
        LinearClass = lora_lib.Linear if self.use_loralib else lora.Linear
        
        for name, module in clip_model.vision_model.named_modules():
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = clip_model.vision_model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
            
                lora_layer = LinearClass(
                    module.in_features, 
                    module.out_features, 
                    r=4,
                    lora_alpha=16,
                    lora_dropout=0,
                    merge_weights=False
                )
      
                lora_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora_layer.bias.data.copy_(module.bias.data)
                
                setattr(parent, child_name, lora_layer)

        for name, param in clip_model.vision_model.named_parameters():
            if 'lora_' not in name: 
                param.requires_grad = False
        
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        loss = self.loss_func(pred, label)

        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)
        
        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
