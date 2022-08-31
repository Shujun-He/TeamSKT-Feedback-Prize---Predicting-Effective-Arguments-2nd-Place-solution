# based on https://www.kaggle.com/code/tascj0/a-text-span-detector

from os.path import join as opj
import re

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from transformers import (AutoModelForTokenClassification, AutoModel, AutoTokenizer,
                          AutoConfig, AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from public_metric_2021 import score_feedback_comp


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
                 scheduler='cosine',
                 num_cycles=1,
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
        self.scheduler = scheduler
        self.num_cycles = num_cycles

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
        self._init_weights(self.head)
    
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
            
    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        logits = self.transformer(input_ids=data['input_ids'],
                                  attention_mask=data['attention_mask']).last_hidden_state
        logits = self.head(logits)
        logits = aggregate_tokens_to_words(logits, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]
        return obj_pred, reg_pred, cls_pred

    def predict(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return dict(text_id=data['text_id'])

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(text_id=data['text_id'],
                    start=to_np(start),
                    end=to_np(end),
                    score=to_np(pr_score),
                    label=to_np(pr_label))
    
    
    def training_step(self, data, trn_df, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_target, reg_target, cls_target, pos_loc = self.build_target(data['gt_spans'], obj_pred, reg_pred, cls_pred)
        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred, cls_pred,
                                                       obj_target, reg_target, cls_target,
                                                       pos_loc)
        loss = obj_loss + reg_loss + cls_loss
        
        res = self.predict(data, test_score_thr)
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for c,start,end in zip(res['label'],res['start'],res['end']):
                pred.append([text_id, LABEL2TYPE[c], ' '.join(np.arange(start,end+1).astype(str))])
            pred_df = pd.DataFrame(pred, columns=['id','class','predictionstring'])
            gt_df = trn_df[trn_df['id']==res['text_id']].reset_index(drop=True)
            score = score_feedback_comp(pred_df, gt_df, return_class_scores=False)
        else:
            score = 0
            
        return {
            'loss':loss,
            'obj_loss':obj_loss.item(),
            'reg_loss':reg_loss.item(),
            'cls_loss':cls_loss.item(),
            'score':score,
        }
    
    def validation_step(self, data, val_df, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_target, reg_target, cls_target, pos_loc = self.build_target(data['gt_spans'], obj_pred, reg_pred, cls_pred)
        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred, cls_pred,
                                                       obj_target, reg_target, cls_target,
                                                       pos_loc)
        
        loss = obj_loss + reg_loss + cls_loss
        
        res = self.predict(data, test_score_thr)
        gt_df = val_df[val_df['id']==res['text_id']].reset_index(drop=True)
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for c,start,end in zip(res['label'],res['start'],res['end']):
                pred.append([text_id, LABEL2TYPE[c], ' '.join(np.arange(start,end+1).astype(str))])
            pred_df = pd.DataFrame(pred, columns=['id','class','predictionstring'])
        else:
            pred_df = pd.DataFrame([], columns=['id','class','predictionstring'])
            
        return {
            'data_id':res['text_id'],
            'loss':loss.item(),
            'obj_loss':obj_loss.item(),
            'reg_loss':reg_loss.item(),
            'cls_loss':cls_loss.item(),
            'pred_df':pred_df,
            'gt_df':gt_df,
        }
    
    def validation_epoch_end(self, outputs, add_epoch=True):
        losses = []
        obj_losses = []
        reg_losses = []
        cls_losses = []
        pred_dfs = []
        gt_dfs = []
        for o in outputs:
            losses.append(o['loss'])
            obj_losses.append(o['obj_loss'])
            reg_losses.append(o['reg_loss'])
            cls_losses.append(o['cls_loss'])
            pred_dfs.append(o['pred_df'])
            gt_dfs.append(o['gt_df'])
        loss = np.array(losses).mean()
        obj_loss = np.array(obj_losses).mean()
        reg_loss = np.array(reg_losses).mean()
        cls_loss = np.array(cls_losses).mean()
        
        pred_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)
        gt_df = pd.concat(gt_dfs, axis=0).reset_index(drop=True)
        score = score_feedback_comp(pred_df, gt_df, return_class_scores=False)
            
        if add_epoch:
            self._current_epoch += 1
            
        return {
            'loss':loss,
            'obj_loss':obj_loss,
            'reg_loss':reg_loss,
            'cls_loss':cls_loss,
            'score':score,
        }
                
    
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
            },
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
    
    def get_losses(self, 
                   obj_pred, reg_pred, cls_pred, 
                   obj_target, reg_target, cls_target, 
                   pos_loc):
        
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        reg_pred = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union
        
        reg_loss = -iou.log().sum() / num_total_samples
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, 
            obj_target,
            reduction='sum') / num_total_samples
        
        return obj_loss, reg_loss, cls_loss

    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        
        # dynamically assign one more positive
        if self.dynamic_positive:
            cls_prob = (obj_pred.sigmoid().unsqueeze(1) *
                        cls_pred.sigmoid()).sqrt()
            for start, end, label in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                cls_cost = F.binary_cross_entropy(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)
                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
            pos_loc = (obj_target == 1).nonzero().flatten()
        return obj_target, reg_target, cls_target, pos_loc
    


class DatasetTrain(Dataset):
    def __init__(self,
                 df,
                 text_dir,
                 tokenizer,
                 mask_prob=0.0,
                 mask_ratio=0.0,
                ):
        self.df = df
        self.samples = list(self.df.groupby('id'))
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

        assert 0 <= mask_prob <= 1
        assert 0 <= mask_ratio <= 1
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]
        
        text_path = opj(self.text_dir, f'{text_id}.txt')
        with open(text_path) as f:
            text = f.read().rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words (offset_mapping = character slices of tokens)
        woff = self.get_word_offsets(text) # (num_words,2) : character slices of words
        toff = offset_mapping # (num_tokens,2) : character slices of tokens
        wx1, wx2 = woff.T # (num_words,) x 2
        tx1, tx2 = toff.T # (num_tokens,) x 2
        ix1 = np.maximum(wx1[..., None], tx1[None, ...]) # (num_words,num_tokens)
        ix2 = np.minimum(wx2[..., None], tx2[None, ...]) # (num_words,num_tokens)
        ux1 = np.minimum(wx1[..., None], tx1[None, ...]) # (num_words,num_tokens)
        ux2 = np.maximum(wx2[..., None], tx2[None, ...]) # (num_words,num_tokens)
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12) # (num_words,num_tokens)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious: # ious.shape=(num_words,num_tokens)
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        # word slices of ground truth spans
        gt_spans = []
        for _, row in df.iterrows():
            winds = row['predictionstring'].split()
            start = int(winds[0])
            end = int(winds[-1])
            span_label = TYPE2LABEL[row['discourse_type']]
            gt_spans.append([start, end + 1, span_label])
        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(
            text=text,
            text_id=text_id,
            input_ids=input_ids, # token ids
            attention_mask=attention_mask,
            word_boxes=word_boxes, # token slices of words
            gt_spans=gt_spans # word slices of ground truth spans
        )

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class CustomCollator(object):
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
#         if self.attention_window is not None:
#             attention_window = self.attention_window
#             padded_length = (attention_window -
#                              max_seq_length % attention_window
#                              ) % attention_window + max_seq_length
#         else:
#             padded_length = max_seq_length
        padded_length = max_seq_length
        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)
    
    
    
class DatasetTest(Dataset):
    def __init__(self, csv_file, text_dir, tokenizer):
        self.df = pd.read_csv(csv_file)
        self.samples = sorted(self.df['id'].unique())
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id = self.samples[index]
        text_path = opj(self.text_dir, f'{text_id}.txt')

        with open(text_path) as f:
            text = f.read().rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        return dict(text=text,
                    text_id=text_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes)

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class TestCollator(object):
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
#         if self.attention_window is not None:
#             attention_window = self.attention_window
#             padded_length = (attention_window -
#                              max_seq_length % attention_window
#                              ) % attention_window + max_seq_length
#         else:
#             padded_length = max_seq_length
        padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes)