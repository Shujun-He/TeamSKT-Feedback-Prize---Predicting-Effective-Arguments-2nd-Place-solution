import torch
print(torch.__name__, torch.__version__)

import argparse
import os
from os.path import join as opj
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_path", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1, required=True)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-effectiveness/', required=False)
    
    parser.add_argument("--test_batch_size", type=int, default=1, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    
    parser.add_argument("--pretrain_path", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    
    parser.add_argument("--num_labels", type=int, default=3, required=False)
    parser.add_argument("--num_labels_2", type=int, default=7, required=False)
    
    parser.add_argument("--l2norm", type=str, default='false', required=False)
    
    parser.add_argument("--max_length", type=int, default=1024, required=False)
    parser.add_argument("--preprocessed_data_path", type=str, required=False)
    
    parser.add_argument("--mt", type=str, default='false', required=False)
    
    parser.add_argument("--window_size", type=int, default=512, required=False)
    parser.add_argument("--inner_len", type=int, default=384, required=False)
    parser.add_argument("--edge_len", type=int, default=64, required=False)
    
    parser.add_argument("--unlabeled_data_path", type=str, required=False)
    
    return parser.parse_args()

    
from models import Model, DatasetTest, CustomCollator
    
if __name__=='__main__':
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
        
    test_df = pd.read_csv(args.unlabeled_data_path)
    print('test_df.shape = ', test_df.shape)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from preprocessing import relation_mapper
    if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in list(relation_mapper.keys())]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in list(relation_mapper.keys())]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    
    from torch.utils.data import DataLoader
    test_dataset = DatasetTest(
        test_df,
        tokenizer, 
    )
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
    
    #model
    model_pretraining = None
    model = Model(args.model, 
                  tokenizer,
                  num_labels=args.num_labels, 
                  num_labels_2=args.num_labels_2,
                  rnn=args.rnn,
                  loss=args.loss,
                  head=args.head,
                  multi_layers=args.multi_layers,
                  l2norm=args.l2norm,
                  max_length=args.max_length,
                  mt=args.mt,
                  window_size=args.window_size,
                  inner_len=args.inner_len,
                  edge_len=args.edge_len,
                  model_pretraining=model_pretraining,
                 )
    weight_path = f'./result/{args.version}/model_seed{args.seed}_fold{args.fold}_swa.pth'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    
    from tqdm import tqdm
    outputs = []
    for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.no_grad():
            output = model.test_step(batch)
            outputs.append(output)
            
    preds = []
    discourse_ids = []
    for o in outputs:
        preds.append(o['pred'])
        discourse_ids.extend(o['discourse_ids'])
    preds = np.vstack(preds)
    discourse_ids = np.hstack(discourse_ids)
    
    pred_df = pd.DataFrame()
    pred_df['discourse_id'] = discourse_ids
    pred_df['Ineffective'] = preds[:,0]
    pred_df['Adequate'] = preds[:,1]
    pred_df['Effective'] = preds[:,2]
    pred_df = test_df.merge(pred_df, on='discourse_id', how='left')
    pred_df.to_csv(f'./result/{args.version}/pseudo_fold{args.fold}.csv', index=False)