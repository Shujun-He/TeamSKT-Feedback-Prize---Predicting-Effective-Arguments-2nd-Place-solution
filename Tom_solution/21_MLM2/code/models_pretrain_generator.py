import torch

def to_gpu(data):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoTokenizer, AdamW, 
                          get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, 
                          get_linear_schedule_with_warmup)
# import sys
# sys.path.append('../../../../../COCO-LM-main/huggingface')
# from cocolm.modeling_cocolm import COCOLMModel
# from cocolm.configuration_cocolm import COCOLMConfig
from sklearn.metrics import f1_score


class LSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                             hidden_size=out_channels,
                             num_layers=num_layers,
                             dropout=p_drop,
                             batch_first=True, 
                             bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
class GRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.GRU(input_size=in_channels,
                           hidden_size=out_channels,
                           num_layers=num_layers,
                           dropout=p_drop,
                           batch_first=True, 
                           bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_layers=1, nhead=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=in_channels,nhead=nhead),
                                                 num_layers=num_layers)
    def forward(self, x):
        x = self.transformer(x)
        return x
    
        
class Model(nn.Module):
    def __init__(self, 
                 model_name, 
                 tokenizer,
                 num_labels, 
                 hidden_dropout_prob=0, #0.1, 
                 learning_rate=1e-5,
                 head_learning_rate=1e-3,
                 num_train_steps=0,
                 p_drop=0, #0.5,
                 warmup_ratio = 0,
                 model_pretraining=None,
                 freeze_layers='false',
                 scheduler='cosine',
                 num_cycles=1,
                 with_cp=False,
                 window_size=-100,
                 inner_len=-100,
                 edge_len=-100,
                 **kwargs,
                ):
        super().__init__()
        self._current_epoch = 1
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_ratio = warmup_ratio 
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.scheduler = scheduler
        self.num_cycles = num_cycles
        
        self.window_size = window_size
        self.inner_len = inner_len
        self.edge_len = edge_len
        
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
        
        # freeze some layers for large models
        if freeze_layers == 'true':
            if 'deberta-v2-xxlarge' in model_name:
                print('freeze 24/48')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:24].requires_grad_(False) # freeze 24/48
            elif 'deberta-v2-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
            elif 'deberta-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
    #         elif 'funnel-transformer-xlarge' in model_name:
    #             self.transformer.embeddings.requires_grad_(False)
    #             self.transformer.encoder.blocks[:1].requires_grad_(False) # freeze 1/3
    
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(self.config.hidden_size, self.num_labels)
        )
        self._init_weights(self.head)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward_logits(self, input_ids, attention_mask):
        #assert self.multi_layers==1
        
        # sliding window approach to deal with longer tokens than max_length
        # https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313235
        L = input_ids.size(1)
        if self.window_size==-100 or L<=self.window_size:
            x = self.transformer(input_ids=input_ids, 
                                 attention_mask=attention_mask).last_hidden_state # (bs,num_tokens,hidden_size)
        else:
            assert len(input_ids)==1
            segments = (L - self.window_size) // self.inner_len
            if (L - self.window_size) % self.inner_len > self.edge_len:
                segments += 1
            elif segments == 0:
                segments += 1
            x = self.transformer(input_ids=input_ids[:,:self.window_size],
                                 attention_mask=attention_mask[:,:self.window_size]).last_hidden_state
            for i in range(1,segments+1):
                start = self.window_size - self.edge_len + (i-1)*self.inner_len
                end   = self.window_size - self.edge_len + (i-1)*self.inner_len + self.window_size
                end = min(end, L)
                x_next = self.transformer(input_ids=input_ids[:,start:end],
                                          attention_mask=attention_mask[:,start:end]).last_hidden_state
                if i==segments:
                    x_next = x_next[:,self.edge_len:]
                else:
                    x_next = x_next[:,self.edge_len:self.edge_len+self.inner_len]
                x = torch.cat([x,x_next], dim=1)
             
        logits = self.head(x) # (bs=1,num_tokens,num_labels)
        return logits
    
    def training_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
        }
        logits = self.forward_logits(**input_data)
        targets = data['mlm_label']
        #loss = self.get_losses(logits.reshape(-1,self.num_labels), targets.reshape(-1,).long()).mean()
        mask = targets!=-100
        #score = self.get_score(logits[mask].sigmoid().detach().cpu().numpy().argmax(-1).reshape(-1,),
        #                       targets[mask].detach().cpu().numpy().reshape(-1,))
        loss = self.get_losses(logits[mask].reshape(-1,self.num_labels),
                               targets[mask].reshape(-1,).long()).mean()
        score = self.get_score(logits.sigmoid().argmax(-1)[mask].detach().cpu().numpy().reshape(-1,),
                               targets[mask].detach().cpu().numpy().reshape(-1,))
        
        #print('logits.sigmoid().argmax(-1)[mask] = ', logits.sigmoid().argmax(-1)[mask])
        #print('targets[mask] = ', targets[mask])
        
        pred = logits.sigmoid().argmax(-1)
        bs = pred.shape[0]
        
        input_ids = data['input_ids']
        orig_input_ids = data['orig_input_ids']
        
        #print('\n')
        #print('pred = ', pred)
        #print('input_ids = ', input_ids)
        #print('targets = ', targets)
        
#         rtd_label = []
#         output_ids = []
#         #print('self.tokenizer.mask_token_id = ', self.tokenizer.mask_token_id)
#         for p,t,inp, orig_inp in zip(pred.reshape(-1,), targets.reshape(-1,), input_ids.reshape(-1,), orig_input_ids.reshape(-1,)):
#             #print('orig_inp = {}, inp = {}, t = {}, p = {}'.format(orig_inp, inp, t, p))
#             if inp==self.tokenizer.mask_token_id:
#                 output_ids.append(p)
#                 rtd_label.append(int(p!=t))
#             elif inp==self.tokenizer.pad_token_id:
#                 output_ids.append(self.tokenizer.pad_token_id)
#                 rtd_label.append(-100)
#             else:
#                 output_ids.append(orig_inp)
#                 rtd_label.append(0)
#         rtd_label = torch.Tensor(rtd_label).reshape(bs,-1)
#         output_ids = torch.Tensor(output_ids).reshape(bs,-1)
        
        #print('rtd_label.shape = ', rtd_label.shape)
        #print('input_ids.shape = ', input_ids.shape)
        
        output_data = data
        #output_data['input_ids'] = output_ids.long().detach()
        #output_data.update({
        #    'rtd_label':rtd_label.long()
        #})
        
        return loss, score, output_data
    
    def validation_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
        }
        logits = self.forward_logits(**input_data)
        targets = data['mlm_label']
        mask = targets!=-100
        #loss = self.get_losses(logits.reshape(-1,self.num_labels), targets.reshape(-1,).long())
        loss = self.get_losses(logits[mask].reshape(-1,self.num_labels), 
                               targets[mask].reshape(-1,).long())
        outputs = {
            'loss':loss.detach().cpu().numpy(),
            'pred':logits.sigmoid().detach().cpu().numpy().argmax(-1),
            'target':targets.detach().cpu().numpy(), 
            'text_list':data['text'],
            'data_id_list':data['data_id']
        }
        
        pred = logits.sigmoid().argmax(-1)
        bs = pred.shape[0]
        
        input_ids = data['input_ids']
        orig_input_ids = data['orig_input_ids']
        
#         rtd_label = []
#         output_ids = []
#         #print('self.tokenizer.mask_token_id = ', self.tokenizer.mask_token_id)
#         for p,t,inp, orig_inp in zip(pred.reshape(-1,), targets.reshape(-1,), input_ids.reshape(-1,), orig_input_ids.reshape(-1,)):
#             #print('orig_inp = {}, inp = {}, t = {}, p = {}'.format(orig_inp, inp, t, p))
#             if inp==self.tokenizer.mask_token_id:
#                 output_ids.append(p)
#                 rtd_label.append(int(p!=t))
#             elif inp==self.tokenizer.pad_token_id:
#                 output_ids.append(self.tokenizer.pad_token_id)
#                 rtd_label.append(-100)
#             else:
#                 output_ids.append(orig_inp)
#                 rtd_label.append(0)
#         rtd_label = torch.Tensor(rtd_label).reshape(bs,-1)
#         output_ids = torch.Tensor(output_ids).reshape(bs,-1)
        
        output_data = data
        #output_data['input_ids'] = output_ids.long()
        #output_data.update({
        #    'rtd_label':rtd_label.long()
        #})
        
        return outputs, output_data
    
    def validation_epoch_end(self, outputs):
        losses = []
        preds = []
        targets = []
        for o in outputs:
            losses.append(o['loss'].reshape(-1,))
            preds.append(o['pred'].reshape(-1,))
            targets.append(o['target'].reshape(-1,))
        loss = np.hstack(losses).mean()
        preds = np.hstack(preds)
        targets = np.hstack(targets)
        
        mask = (targets!=-100)
        preds = preds[mask]
        targets = targets[mask]
        score = self.get_score(preds, targets)
        self._current_epoch += 1
        
        return loss, score
    
    
    def configure_optimizers(self):
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler(optimizer)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def fetch_optimizer(self):
        head_params = list(self.head.named_parameters())
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n,p in head_params], 
                "weight_decay": 0.01,
                "lr": self.head_learning_rate,
            }
        ]
        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        return optimizer
    
    def fetch_scheduler(self, optimizer):
        print('self.warmup_ratio = ', self.warmup_ratio)
        print('self.num_train_steps = ', self.num_train_steps)
        print('num_warmup_steps = ', int(self.warmup_ratio * self.num_train_steps))
        if self.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=0.5*self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='cosine_hard':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                last_epoch=-1,
            )
        return scheduler
    
    def get_losses(self, logits, target):
        loss = self.loss_fn(logits, target)
        return loss
    
    def get_score(self, pred, target):
        score = f1_score(target, pred, average='micro')
        return score
    
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re
from tqdm import tqdm
from os.path import join as opj

class DatasetTrain(Dataset):
    def __init__(self, df, tokenizer, ratio_masking, max_length, text_dir, mode='train'):
        self.df = df
        self.unique_ids = sorted(df['essay_id'].unique())
        self.tokenizer = tokenizer
        self.ratio_masking = ratio_masking
        self.text_dir = text_dir
        self.max_length = max_length
        
        self.mode = mode
        self.idxs_masking_list = []
        if self.mode=='valid':
            print('create validation dataset...')
            for essay_id in tqdm(self.unique_ids):
                sample_df = self.df[self.df['essay_id']==essay_id].reset_index(drop=True)
                #text = ''
                #for discourse_type, discourse_text in zip(sample_df['discourse_type'].values, sample_df['discourse_text'].values):
                #    text += f' [{discourse_type.upper()}] {discourse_text}'
                text_path = opj(self.text_dir, f'{essay_id}.txt')
                with open(text_path) as f:
                    text = f.read().rstrip()
                    
                if self.max_length!=-100:
                    # random crop words
                    words = text.split()
                    start = np.random.randint(0,max(1,len(words)-self.max_length)) # approximation
                    words = words[start:]
                    text = ' '.join(words)
                    tokens = self.tokenizer.encode_plus(text, add_special_tokens=False, max_length=self.max_length-2, truncation=True)
                else:
                    tokens = self.tokenizer.encode_plus(text, add_special_tokens=False)
                    
                input_ids = tokens['input_ids']
                
                num_tokens = len(input_ids)
                num_masking = int(num_tokens * self.ratio_masking)
                idxs_masking = np.random.permutation(num_tokens)[:num_masking]
                self.idxs_masking_list.append(idxs_masking)
            print('create validation dataset, done')
        
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        essay_id = self.unique_ids[idx]
        sample_df = self.df[self.df['essay_id']==essay_id].reset_index(drop=True)
        #text = ''
        #for discourse_type, discourse_text in zip(sample_df['discourse_type'].values, sample_df['discourse_text'].values):
        #    text += f' [{discourse_type.upper()}] {discourse_text}'
        text_path = opj(self.text_dir, f'{essay_id}.txt')
        with open(text_path) as f:
            text = f.read().rstrip()
        
        if self.max_length!=-100:
            # random crop words
            words = text.split()
            start = np.random.randint(0,max(1,len(words)-self.max_length)) # approximation
            words = words[start:]
            text = ' '.join(words)
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=False, max_length=self.max_length-2, truncation=True)
        else:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=False)
            
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        orig_input_ids = tokens['input_ids']
        
        # MLM
        if self.mode=='train':
            num_tokens = len(input_ids)
            num_masking = int(num_tokens * self.ratio_masking)
            idxs_masking = np.random.permutation(num_tokens)[:num_masking]
        elif self.mode=='valid':
            idxs_masking = self.idxs_masking_list[idx]
        mlm_label = [input_id if i in idxs_masking else -100 for i,input_id in enumerate(input_ids)]
        input_ids = [input_id if i not in idxs_masking else self.tokenizer.mask_token_id for i,input_id in enumerate(input_ids)]
        
        # add special tokens
        mlm_label = [-100] + mlm_label + [-100]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] + attention_mask + [1]
        orig_input_ids = [self.tokenizer.cls_token_id] + orig_input_ids + [self.tokenizer.sep_token_id]
        
        return dict(
            data_id = essay_id, #data_id,
            text = text,
            mlm_label = torch.LongTensor(mlm_label),
            input_ids = torch.LongTensor(input_ids),
            attention_mask = torch.LongTensor(attention_mask),
            orig_input_ids = torch.LongTensor(orig_input_ids),
        )
            
class CustomCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, samples):
        output = dict()
        
        for k in samples[0].keys():
            output[k] = [sample[k] for sample in samples]
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        
        # add padding
        if self.tokenizer.padding_side == "right":
            output["mlm_label"] = [s.tolist() + (batch_max - len(s)) * [-100] for s in output["mlm_label"]]
            output["input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s.tolist() + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            output["orig_input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["orig_input_ids"]]
        else:
            output["mlm_label"] = [(batch_max - len(s)) * [-100] + s.tolist() for s in output["mlm_label"]]
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist() for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s.tolist() for s in output["attention_mask"]]
            output["orig_input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist() for s in output["orig_input_ids"]]
            
        output["input_ids"] = torch.LongTensor(output["input_ids"])
        output["attention_mask"] = torch.LongTensor(output["attention_mask"])
        output["mlm_label"] = torch.LongTensor(output["mlm_label"])
        output["orig_input_ids"] = torch.LongTensor(output["orig_input_ids"])
        return output