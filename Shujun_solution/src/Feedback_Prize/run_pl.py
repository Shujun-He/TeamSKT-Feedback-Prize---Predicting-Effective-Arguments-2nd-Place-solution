import json
import os
import torch.nn as nn
import torch
import warnings
import argparse
from Logger import *
import pickle
from Dataset import *
warnings.filterwarnings("ignore")
from Functions import *
from Network import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold
import numpy as np, os
from scipy import stats
import pandas as pd, gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from scipy.stats import pearsonr
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import sys
import torch.utils.checkpoint
from DebertaV2Converter import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--input_path', type=str, default='../../input', help='path of input data files')
    parser.add_argument('--downloaded_model_path', type=str, default='../../input/deberta-base', help='path of downloader hugginface model path')
    parser.add_argument('--load_model_path', type=str, default='', help='path of model path to load')
    parser.add_argument('--train_csv_path', type=str, default='', help='path of train csv file')
    parser.add_argument('--pl_label_csv', type=str, default='', help='pl_label_csv')
    parser.add_argument('--train_pl_csv_path', type=str, default='', help='path of train csv file')
    parser.add_argument('--full_text_path', type=str, default='', help='path of train csv file')
    parser.add_argument('--full_pl_text_path', type=str, default='../../input/feedback-prize-2021/train', help='path of train csv file')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-base', help='name of model. this is needed if you haven\'t downloaded your model yet')
    parser.add_argument('--rnn', type=str, default='GRU', help='rnn')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
    parser.add_argument('--pl_epochs', type=int, default=0, help='number of pl epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each batch during training')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='max gradient norm during training before clipping')
    parser.add_argument('--max_len', type=int, default=512,)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--window_size', type=int, default=512, help='window_size')
    parser.add_argument('--lr_schedule', type=str, default="step", help='type of lr schdule')
    parser.add_argument('--nclass', type=int, default=3, help='number of classes from the linear decoder')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr_scale', type=float, default=1, help='learning rate scale')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--arg_prob', type=float, default=0, help='probability to do aug during training')
    #parser.add_argument('--softmax', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--do_val', action='store_true', help='do validation or not')
    parser.add_argument('--gradient_checkpointing_enable', action='store_true', help='use gradient checkpoint')
    parser.add_argument('--loss', default="CrossEntropyLoss", type=str, help='type of loss')
    parser.add_argument('--experiment_name', type=str, default="")
    opts = parser.parse_args()
    return opts


args=get_args()


# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id #0,1,2,3 for four gpu

device='cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using model fils from {args.downloaded_model_path}")

if args.downloaded_model_path is None:
    args.downloaded_model_path = 'model'

#if args.lr_schedule=='step':
learning_rates=[2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]
learning_rates=[lr*args.lr_scale for lr in learning_rates]

print('learning_rates:')

print(learning_rates)


if args.downloaded_model_path == 'model':
    os.system('mkdir model')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(args.model_name)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(args.model_name,
                                                               config=config_model)
    backbone.save_pretrained('model')

#load data and libraries
# train=pd.read_csv(os.path.join(args.input_path, "us-patent-phrase-to-phrase-matching","train.csv"))
# cpc_texts = pd.read_json(os.path.join(args.input_path, "cpc-scheme-dataframe.json"), orient='records')

train=pd.read_csv(args.train_csv_path)#.iloc[:1000]

MEAN_DISCOURSE=train.groupby("essay_id")['discourse_type'].transform("count").mean()

#exit()

label_mapping={'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
discourse_mapping={k:v for v,k in enumerate(train['discourse_type'].unique())}
discourse_mapping={'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3, 'Counterclaim': 4, 'Rebuttal': 5, 'Concluding Statement': 6}



full_texts={}

for essay_id in train['essay_id']:
    with open(os.path.join(args.full_text_path,essay_id+'.txt'),'r') as f:
        text=f.read()

    full_texts[essay_id]=text


train_pl=pd.read_csv(args.train_pl_csv_path)
train_pl=train_pl.rename(columns={'id':"essay_id"})


full_texts_pl={}

for essay_id in train_pl['essay_id']:
    with open(os.path.join(args.full_pl_text_path,essay_id+'.txt'),'r') as f:
        text=f.read()

    full_texts_pl[essay_id]=text
train_pl['discourse_id']=[str(f) for f in train_pl['discourse_id']]

if args.pl_label_csv!='':
    pl_labels=pd.read_csv(args.pl_label_csv)
    pl_labels['discourse_id']=[str(f) for f in pl_labels['discourse_id']]
    pl_labels['label']=[vector for vector in pl_labels[list(label_mapping.keys())].values]


    train_pl=train_pl.merge(pl_labels[['discourse_id','label']],how='left',on='discourse_id')
    train_pl=train_pl[train_pl['label'].isna()==False]



train['label'] = train['discourse_effectiveness'].map(label_mapping)

os.system(f'mkdir {args.experiment_name}')
os.system(f'mkdir {args.experiment_name}/logs')
os.system(f'mkdir {args.experiment_name}/models')
os.system(f'mkdir {args.experiment_name}/oofs')

with open(f'{args.experiment_name}/commandline_args_{args.fold}.txt', 'w+') as f:
    json.dump(args.__dict__, f, indent=2)

train['fold']=-1
#split data
# if args.do_val==False:
#     train['fold'] = [-1 for _ in range(len(train))]
#
# else:
    #StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=args.nfolds,shuffle=True,random_state=2022)
for fold, (train_idxs, test_idxs) in enumerate(cv.split(train, train['label'], train['essay_id'])):
    train['fold'][test_idxs]=fold

for group in train.groupby('essay_id'):
    assert len(group[1]['fold'].unique())==1

print(train.groupby('fold')["discourse_effectiveness"].value_counts())


train_folds = train[train['fold'] != args.fold].reset_index(drop=True)
valid_folds = train[train['fold'] == args.fold].reset_index(drop=True)


#tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)
if "deberta-v3" in args.model_name or "deberta-v2" in args.model_name:
    #tokenizer = DebertaV2TokenizerFast.from_pretrained(args.downloaded_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)
    tokenizer = convert_deberta_v2_tokenizer(tokenizer)
elif 'coco' in args.model_name:
    tokenizer = COCOLMTokenizer.from_pretrained(args.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)


if args.pl_label_csv!='':
    pl_set = FeedbackDataset(tokenizer, train_pl, full_texts_pl, True, args.arg_prob, args.loss,args.max_len)
training_set = FeedbackDataset(tokenizer, train_folds, full_texts, True, args.arg_prob, args.loss,args.max_len)
testing_set = FeedbackDataset(tokenizer, valid_folds, full_texts, False, 0 , args.loss,args.max_len)

#valid_folds.to_csv(f"{args.experiment_name}/oofs/{args.fold}.csv")
#exit()
print(f"Training set has {len(training_set)} samples")
print(f"Val set has {len(testing_set)} samples")
if args.pl_label_csv!='':
    print(f"PL set has {len(pl_set)} samples")

# TRAIN DATASET AND VALID DATASET
train_params = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': args.num_workers,
                'pin_memory':True
                }

test_params = {'batch_size': args.batch_size,
                'shuffle': False,
                'num_workers': args.num_workers,
                'pin_memory':True
                }
if args.pl_label_csv!='':
    pl_loader = DataLoader(pl_set, **train_params, collate_fn=CustomCollate(tokenizer))
else:
    assert args.pl_epochs==0

training_loader = DataLoader(training_set, **train_params, collate_fn=CustomCollate(tokenizer))
testing_loader = DataLoader(testing_set, **test_params, collate_fn=CustomCollate(tokenizer,False))

# for batch in tqdm(pl_loader):
#     pass
# exit()


columns=['epoch','train_loss','val_loss']
logger=CSVLogger(columns,f"{args.experiment_name}/logs/fold{args.fold}.csv")

from torch.cuda.amp import GradScaler

scaler = GradScaler()

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train_one_epoch(epoch,training_loader):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    #tr_preds, tr_labels = [], []
    if args.loss=='CrossEntropyLoss':
        criterion=nn.CrossEntropyLoss(reduction='none')
    elif args.loss=='BCELoss' or args.loss=='OrdinalLoss':
        criterion=nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss=='MSELoss':
        criterion=nn.MSELoss(reduction='none')
    #criterion=DiceLoss(reduction='none')
    #criterion=DiceLoss(square_denominator=True, with_logits=True, index_label_position=True,
                        #smooth=1, ohem_ratio=0, alpha=0.01, reduction="none")
    # put model in training mode
    model.train()
    total_steps=len(training_loader)
    bar=tqdm(enumerate(training_loader),total=total_steps)
    #break_point=len(training_loader)//2
    for idx, batch in bar:


        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        sequence_ids = batch['sequence_ids'].to(device, dtype = torch.long)
        gather_indices = batch['gather_indices'].to(device, dtype = torch.long)
        #gather_indices_by_row = batch['gather_indices_by_row'].to(device, dtype = torch.long)
        discourse_type_ids = batch['discourse_type_ids'].to(device, dtype = torch.long)
        #labels=
        if args.loss=='CrossEntropyLoss':
            if len(batch['labels'].shape)>1:
                labels = batch['labels'].to(device, dtype = torch.float)
            else:
                labels = batch['labels'].to(device, dtype = torch.long)
        else:
            labels = batch['labels'].to(device, dtype = torch.float)

        # print(len(labels))
        # exit()
        with torch.autocast(device_type="cuda"):
 #           fgm.attack()
            output = model(ids,mask, sequence_ids, discourse_type_ids, gather_indices)
            # print(output.shape)
            # exit()


            loss=criterion(output,labels).mean()/args.gradient_accumulation_steps#*len(labels)/(ids.shape[0]*MEAN_DISCOURSE)



        tr_loss += loss.item()*args.gradient_accumulation_steps

        bar.set_postfix({'train_loss': tr_loss/(idx+1)})

        nb_tr_steps += 1
        scaler.scale(loss).backward()



        if idx%args.gradient_accumulation_steps==0 or idx==(total_steps-1):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=args.max_grad_norm
            )
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            if args.lr_schedule=='linear anneal' or args.lr_schedule=='cosine anneal':
                lr_scheduler.step()

        # if idx==break_point:
        #     break
        #break

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    return epoch_loss


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def val(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    #tr_preds, tr_labels = [], []
    if args.loss=='CrossEntropyLoss':
        criterion=nn.CrossEntropyLoss(reduction='none')
    elif args.loss=='BCELoss' or args.loss=='OrdinalLoss':
        criterion=nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss=='MSELoss':
        criterion=nn.MSELoss(reduction='none')
    #criterion=DiceLoss(reduction='none')
    #criterion=DiceLoss(square_denominator=True, with_logits=True, index_label_position=True,
                        #smooth=1, ohem_ratio=0, alpha=0.01, reduction="none")
    # put model in training mode
    model.eval()
    bar=tqdm(enumerate(testing_loader),total=len(testing_loader))
    preds=[]
    pred_prob_sequences=[]
    loss_per_discourse=[]
    discourse_ids=[]
    gt_labels=[]
    #seq_ids=[]
    for idx, batch in bar:


        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        sequence_ids = batch['sequence_ids'].to(device, dtype = torch.long)
        sample_id=batch['sample_id']
        gather_indices = batch['gather_indices'].to(device, dtype = torch.long)
        max_sample_id=sample_id.max()
        discourse_ids=discourse_ids+batch['discourse_ids']
        discourse_type_ids = batch['discourse_type_ids'].to(device, dtype = torch.long)
        # print(ids.shape)
        # print(mask.shape)
        # exit()

        if args.loss=='CrossEntropyLoss':
            labels = batch['labels'].to(device, dtype = torch.long)
        else:
            labels = batch['labels'].to(device, dtype = torch.float)
        #seq_ids = batch['sequence_ids'].to(device, dtype = torch.long)



        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                output, vectors = model(ids,mask,sequence_ids,discourse_type_ids,gather_indices,return_vectors=True)
                vectors=[torch.nn.functional.softmax(v,-1).cpu().numpy() for v in vectors]
                pred_prob_sequences=pred_prob_sequences+vectors
                # print(vectors[0].shape)
                # exit()
                batch_loss=criterion(output,labels)
                loss=batch_loss.mean()#.mean()#/args.gradient_accumulation_steps
                preds.append(torch.nn.functional.softmax(output,-1).cpu())
                gt_labels.append(labels.cpu())
                loss_per_discourse.append(batch_loss.cpu())
        loss/=(max_sample_id+1)





        tr_loss += loss.item()

        bar.set_postfix({'val_loss': tr_loss/(idx+1)})

        nb_tr_steps += 1

    preds=torch.cat(preds).numpy()
    gt_labels=torch.cat(gt_labels).numpy()

    oof=pd.DataFrame(columns=['discourse_id']+list(label_mapping.keys()))
    oof['discourse_id']=discourse_ids
    oof[list(label_mapping.keys())]=preds
    oof['label']=gt_labels
    oof['prob_sequences']=pred_prob_sequences
    # print(oof.head())
    # exit()

    epoch_loss = torch.cat(loss_per_discourse).mean()
    tr_accuracy = 0
    print(f"Validation loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")
    return epoch_loss, oof, pred_prob_sequences



# CREATE MODEL

model = SlidingWindowTransformerModel(args.downloaded_model_path, window_size=args.window_size,nclass=args.nclass,rnn=args.rnn)
model.to(device)

#fgm = FGM(model)


#exit()
lr=learning_rates[0]

#exit()

param_groups=model.parameters()
optimizer = torch.optim.AdamW(params=param_groups, lr=learning_rates[0], weight_decay=args.weight_decay)


#exit()

if args.load_model_path!='':
    model.load_state_dict(torch.load(args.load_model_path))

if args.gradient_checkpointing_enable:
    model.backbone.gradient_checkpointing_enable()

from transformers import get_scheduler, get_cosine_schedule_with_warmup

if args.pl_label_csv!='':
    num_warmup_steps=int((len(training_loader)*(args.epochs-args.pl_epochs)//args.gradient_accumulation_steps+args.pl_epochs*len(pl_loader)//args.gradient_accumulation_steps)*0.1)
    num_training_steps=len(training_loader)*(args.epochs-args.pl_epochs)//args.gradient_accumulation_steps+args.pl_epochs*len(pl_loader)//args.gradient_accumulation_steps
else:
    num_warmup_steps=int((len(training_loader)*(args.epochs-args.pl_epochs)//args.gradient_accumulation_steps)*0.1)
    num_training_steps=len(training_loader)*(args.epochs-args.pl_epochs)//args.gradient_accumulation_steps

if args.lr_schedule=='linear anneal':

    print(f'total_training_steps:{num_training_steps}')
    print(f'warmup_steps:{num_warmup_steps}')

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print("Using linear anneal schedule")

elif args.lr_schedule=='cosine anneal':


    print(f'total_training_steps:{num_training_steps}')
    print(f'warmup_steps:{num_warmup_steps}')

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print("Using cosine anneal schedule")

# print(args.pl_epochs)
# exit()

best_score=10000
val_labels=valid_folds['label'].values
for epoch in range(args.epochs):

    print(f"### Training epoch: {epoch + 1}")
    if args.lr_schedule=='step':
        for i,g in enumerate(optimizer.param_groups):
            g['lr'] = learning_rates[epoch]
    #lr = optimizer.param_groups[0]['lr']
    lr_first_layer=optimizer.param_groups[0]['lr']
    lr_last_layer=optimizer.param_groups[-1]['lr']
    print(f'### first layer LR = {lr_first_layer} last layer LR = {lr_last_layer}\n')

    if epoch<args.pl_epochs:
        train_loss = train_one_epoch(epoch,pl_loader)
    else:
        train_loss = train_one_epoch(epoch,training_loader)
    torch.cuda.empty_cache()
    gc.collect()

    if args.do_val:
        #val_loss, oof = val(epoch)
        val_loss, oof, pred_prob_sequences = val(epoch)


    if val_loss<best_score:
        best_score=val_loss
        torch.save(model.state_dict(), f'{args.experiment_name}/models/fold{args.fold}.pt')
        # with open(f"{args.experiment_name}/oofs/{args.fold}.p",'wb+') as f:
        #     pickle.dump({"val_labels":val_labels,"preds":preds},f)
        with open(f"{args.experiment_name}/oofs/{args.fold}.p",'wb+') as f:
            pickle.dump(pred_prob_sequences,f)
        oof.to_csv(f"{args.experiment_name}/oofs/{args.fold}.csv",index=False)
    elif args.do_val==False:
        torch.save(model.state_dict(), f'{args.experiment_name}/models/fold{args.fold}.pt')

    print()
    print(f'Val loss for epoch {epoch}',val_loss)
    print()

    logger.log([epoch,train_loss,val_loss])
