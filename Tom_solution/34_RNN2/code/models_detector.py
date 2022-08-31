# based on https://www.kaggle.com/code/tascj0/a-text-span-detector

from os.path import join as opj
import re
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from transformers import (AutoModelForTokenClassification, AutoModel, AutoTokenizer,
                          AutoConfig, AdamW, get_cosine_schedule_with_warmup)

LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')
TYPE2LABEL = {t:l for l, t in enumerate(LABEL2TYPE)}
LABEL2TYPE = {l:t for t,l in TYPE2LABEL.items()}


def to_gpu(data):
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep

class TextSpanDetectorOriginal(nn.Module):
    def __init__(self, arch, num_classes=7, local_files_only=True):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            arch,
            num_labels=1 + 2 + num_classes,
            local_files_only=local_files_only
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            arch, 
            local_files_only=local_files_only
        )
                        
class TextSpanDetector(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_classes=7,
                 dynamic_positive=False,
                 with_cp=False,
                 hidden_dropout_prob=0, 
                 learning_rate=1e-5,
                 head_learning_rate=1e-3,
                 num_train_steps=0,
                 p_drop=0,
                 warmup_ratio = 0,
                 model_pretraining=None,
                 rnn='none',
                 loss='mse',
                 head='simple',
                 msd='false',
                 multi_layers=1,
                 aug='none',
                 mixup_alpha=1.0,
                 aug_stop_epoch=999,
                 p_aug=0,
                 adv_sift='false',
                 l2norm='false',
                 s=30,
                 weight_decay=0.01,
                 freeze_layers='false',
                 mt='false',
                 w_mt=1,
                ):
        super().__init__()
        self._current_epoch = 1
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_ratio = warmup_ratio 
        self.num_train_steps = num_train_steps
        self.num_labels = 1 + 2 + num_classes
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.loss = loss
        self.msd = msd
        self.multi_layers = multi_layers
        self.aug = aug
        self.mixup_alpha = mixup_alpha
        self.aug_stop_epoch = aug_stop_epoch
        self.p_aug = p_aug
        self.adv_sift = adv_sift
        self.l2norm = l2norm
        self.s = s
        self.weight_decay = weight_decay
        self.mt = mt
        self.w_mt = w_mt

        self.num_classes = num_classes
        self.dynamic_positive = dynamic_positive
        
        if model_pretraining is not None:
            self.transformer = model_pretraining.transformer
            self.config = model_pretraining.config
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.update(
                {
                    "output_hidden_states": True,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "add_pooling_layer": False,
                    "num_labels": self.num_labels,
                }
            )
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
            
        # resize
        self.transformer.resize_token_embeddings(len(tokenizer))
        
        if with_cp:
            self.transformer.gradient_checkpointing_enable()
            
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(self.config.hidden_size, self.num_labels)
        )