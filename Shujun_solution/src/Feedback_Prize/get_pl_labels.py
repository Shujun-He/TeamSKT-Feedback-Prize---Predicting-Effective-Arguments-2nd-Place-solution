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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
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
from DebertaV2Converter import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--input_path', type=str, default='../../input', help='path of input data files')
    parser.add_argument('--downloaded_model_path', type=str, default='../../input/deberta-base', help='path of downloader hugginface model path')
    parser.add_argument('--load_model_path', type=str, default='', help='path of model path to load')
    parser.add_argument('--train_csv_path', type=str, default='', help='path of train csv file')
    parser.add_argument('--full_text_path', type=str, default='', help='path of train csv file')
    parser.add_argument('--cpc_path', type=str, default='', help='path of cpc csv file')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-base', help='name of model. this is needed if you haven\'t downloaded your model yet')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
    parser.add_argument('--rnn', type=str, default='GRU', help='rnn')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each batch during training')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='max gradient norm during training before clipping')
    parser.add_argument('--max_len', type=int, default=512,)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--hidden_state_dimension', type=int, default=768, help='size of hidden state outputter by transformer encoder')
    parser.add_argument('--window_size', type=int, default=512, help='window_size')
    parser.add_argument('--lr_schedule', type=str, default="step", help='type of lr schdule')
    parser.add_argument('--nclass', type=int, default=1, help='number of classes from the linear decoder')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr_scale', type=float, default=1, help='learning rate scale')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--layerwise_lr_decay_rate', type=float, default=1, help='layerwise lr decay rate')
    parser.add_argument('--arg_prob', type=float, default=0, help='probability to do aug during training')
    #parser.add_argument('--softmax', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--do_val', action='store_true', help='do validation or not')
    parser.add_argument('--loss', type=str, help='type of loss')
    parser.add_argument('--split_by', default="anchor", type=str, help='split by anchor/score')
    parser.add_argument('--cpc_level', type=int, default=4)
    parser.add_argument('--experiment_name', type=str, default="")
    #parser.add_argument('--convert_chemical_foruma', type=str, default="")
    parser.add_argument('--convert_chemical_formulas', action='store_true', help='convert chemical formulas or not')
    opts = parser.parse_args()
    return opts



#def train_fold():

args=get_args()

args.nclass=3


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

train=pd.read_csv(args.train_csv_path)#.iloc[:2000]

train=train.rename(columns={'id':"essay_id"})

full_texts={}

for essay_id in train['essay_id'].unique():
    with open(os.path.join(args.full_text_path,essay_id+'.txt'),'r') as f:
        text=f.read()

    full_texts[essay_id]=text

train['discourse_id']=[str(f) for f in train['discourse_id']]

train['label']=-1

#exit()

#exit()


#exit()


#cpc_texts = torch.load(os.path.join(args.input_path,"cpc_texts.pth"))
#train['context_text'] = train['context'].map(cpc_texts)
#train['text'] = '[CLS]'+ train['anchor'] + '[SEP]' + train['target'] + '[SEP]'  + train['context_text']
#exit()

label_mapping={'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
discourse_mapping={k:v for v,k in enumerate(train['discourse_type'].unique())}
discourse_mapping={'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3, 'Counterclaim': 4, 'Rebuttal': 5, 'Concluding Statement': 6}

# print(discourse_mapping)
# exit()
#train['label'] = train['discourse_effectiveness'].map(label_mapping)

os.system(f'mkdir {args.experiment_name}')
os.system(f'mkdir {args.experiment_name}/logs')
os.system(f'mkdir {args.experiment_name}/models')
os.system(f'mkdir {args.experiment_name}/oofs')
os.system(f'mkdir {args.experiment_name}/pl_labels')




#tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)
if "deberta-v3" in args.model_name or "deberta-v2" in args.model_name:
    #tokenizer = DebertaV2TokenizerFast.from_pretrained(args.downloaded_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)
    tokenizer = convert_deberta_v2_tokenizer(tokenizer)
elif 'coco' in args.model_name:
    tokenizer = COCOLMTokenizer.from_pretrained(args.model_name)

else:
    tokenizer = AutoTokenizer.from_pretrained(args.downloaded_model_path)


#training_set = FeedbackDataset(tokenizer, train_folds, full_texts, True, args.arg_prob, args.loss,args.max_len)
testing_set = FeedbackDataset(tokenizer, train, full_texts, False, 0 , args.loss,args.max_len)

#exit()

#exit()
print(f"Pl set has {len(testing_set)} samples")


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

testing_loader = DataLoader(testing_set, **test_params, collate_fn=CustomCollate(tokenizer,False))



from torch.cuda.amp import GradScaler

scaler = GradScaler()


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
    loss_per_discourse=[]
    discourse_ids=[]
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
                output = model(ids,mask,sequence_ids,discourse_type_ids,gather_indices)
                # print(output.shape)
                # print(labels.shape)
                # print(labels)
                # exit()
                #batch_loss=criterion(output,labels)
                #loss=batch_loss.mean()#.mean()#/args.gradient_accumulation_steps
                preds.append(torch.nn.functional.softmax(output,-1).cpu())
                #loss_per_discourse.append(batch_loss.cpu())
                #loss/=(max_sample_id+1)





        #tr_loss += loss.item()

        bar.set_postfix({'val_loss': tr_loss/(idx+1)})

        nb_tr_steps += 1

    preds=torch.cat(preds).numpy()

    # print(preds.shape)
    # print(len(discourse_ids))
    # exit()
    #preds=torch.cat(preds).cpu().numpy()
    # if args.loss=='CrossEntropyLoss':
    #     preds=torch.stack(preds).cpu().numpy()
    # elif args.loss=='OrdinalLoss':
    #     preds=torch.stack(preds).cpu().numpy()
    #     #preds=preds.sum(-1)*0.25
    #     # print(preds.shape)
    #     # exit()
    # else:
    #     preds=torch.tensor(preds).cpu().numpy()
    oof=pd.DataFrame(columns=['discourse_id']+list(label_mapping.keys()))
    oof['discourse_id']=discourse_ids
    oof[list(label_mapping.keys())]=preds

    # print(oof.head())
    # exit()
    #print(f"Training accuracy epoch: {tr_accuracy}")
    return oof



# CREATE MODEL

model = SlidingWindowTransformerModel(args.downloaded_model_path, window_size=args.window_size,nclass=args.nclass,rnn=args.rnn)

model.to(device)



#exit
#{'params': model.classifier.parameters(), 'lr': 1e-3}



#exit()


#exit()

if args.load_model_path!='':
    model.load_state_dict(torch.load(args.load_model_path))




model.load_state_dict(torch.load(f'{args.experiment_name}/models/fold{args.fold}.pt'))


pl_labels = val(0)
pl_labels.to_csv(f"{args.experiment_name}/pl_labels/{args.fold}.csv",index=False)
#valid_texts = valid_folds['text'].values
#features_texts = valid_folds['feature_text'].values
#valid_labels = create_labels_for_scoring(valid_folds)
# best_score=10000
# val_labels=valid_folds['label'].values
# for epoch in range(args.epochs):
#
#     print(f"### Training epoch: {epoch + 1}")
#     if args.lr_schedule=='step':
#         for i,g in enumerate(optimizer.param_groups):
#
#             g['lr'] = learning_rates[epoch]*args.layerwise_lr_decay_rate**(i)
#     #lr = optimizer.param_groups[0]['lr']
#     lr_first_layer=optimizer.param_groups[0]['lr']
#     lr_last_layer=optimizer.param_groups[-1]['lr']
#     print(f'### first layer LR = {lr_first_layer} last layer LR = {lr_last_layer}\n')
#
#     train_loss = train_one_epoch(epoch)
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     if args.do_val:
#         val_loss, oof = val(epoch)
#
#
#     if val_loss<best_score:
#         best_score=val_loss
#         torch.save(model.state_dict(), f'{args.experiment_name}/models/fold{args.fold}.pt')
#         # with open(f"{args.experiment_name}/oofs/{args.fold}.p",'wb+') as f:
#         #     pickle.dump({"val_labels":val_labels,"preds":preds},f)
#         oof.to_csv(f"{args.experiment_name}/oofs/{args.fold}.csv",index=False)
#     elif args.do_val==False:
#         torch.save(model.state_dict(), f'{args.experiment_name}/models/fold{args.fold}.pt')
#
#     print()
#     print(f'Val loss for epoch {epoch}',val_loss)
#     print()
#
#     logger.log([epoch,train_loss,val_loss])
