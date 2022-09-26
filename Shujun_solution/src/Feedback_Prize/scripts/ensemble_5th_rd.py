#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import log_loss
from tqdm import tqdm
import os

# In[2]:


train_kaggle=pd.read_csv("../../input/feedback-prize-effectiveness/train.csv")
target='label'
label_mapping={'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
train_kaggle['label'] = train_kaggle['discourse_effectiveness'].map(label_mapping)
train_kaggle


# In[3]:


len(train_kaggle['discourse_id'].unique())


# In[21]:


save_path="./gbm_models"


exps=["deberta_v3_large_pl_5th_tascj0",'deberta_v2_xlarge_pl_5th_tascj0',
'deberta_xlarge_pl_5th_tascj0',
'deberta_v2_xlarge_pl_5th','deberta_v3_large_pl_5th']

folder='ensemble_pl_labels_1st_rd'

os.system(f'mkdir {folder}')


#exps+=["deberta_v3_large_pl_3rd_tascj0","deberta_v2_xlarge_pl_3rd_tascj0"]
#exps=["longformer_large_pl"]
ensemble_predictions=[]
for exp in exps:
    #exp="deberta_v2_xlarge_pl"
    oof_path="."
    NFOLDS=6
    train_df=[]
    prob_sequences=[]
    print(exp)
    for fold in range(NFOLDS):
        df=pd.read_csv(f"{exp}/{oof_path}/oofs/{fold}.csv")
        df['kfold']=fold
        train_df.append(df)
        with open(f"{exp}/{oof_path}/oofs/{fold}.p",'rb') as f:
            prob_sequences+=pickle.load(f)

    y_oof=np.load(f"{exp}/nn_oofs.npy")

    train_df=pd.concat(train_df)

    #print(train_df)

    if 'discourse_type' not in train_df.columns:
        train_df=train_df.merge(train_kaggle[['discourse_id','discourse_type','essay_id']],how='left',on='discourse_id')
    else:
        train_df=train_df.merge(train_kaggle[['discourse_id','essay_id']],how='left',on='discourse_id')
    train_df['discourse_type']=train_df['discourse_type'].astype("category")

    train_df[['Ineffective','Adequate','Effective',]]=y_oof

    preds=train_kaggle.merge(train_df[['discourse_id','Ineffective','Adequate','Effective',]],on='discourse_id',how='left')

    preds[['Ineffective','Adequate','Effective',]]=preds[['Ineffective','Adequate','Effective',]].fillna(1e-9)
    preds
    ensemble_predictions.append(preds[['Ineffective','Adequate','Effective',]].values)
    print(f"for {exp}: val_loss is {log_loss(preds[target],preds[['Ineffective','Adequate','Effective']])}")
    #train_df

ensemble_predictions=np.stack(ensemble_predictions)#.mean(0)


# In[22]:


def ensemble_loss(weights,return_pred=False):
    weights=np.array(weights)
    weights=weights.reshape(-1,1,1)/weights.sum()
    #print(weights)
    p=weights.reshape(-1,1,1)*ensemble_predictions
    p=p.sum(0)
    loss=log_loss(preds[target],p)
    if return_pred:
        return loss,p
    else:
        return loss

#ensemble_loss(np.array([45,55]))



# In[31]:


from skopt import gp_minimize

results=gp_minimize(ensemble_loss, np.array([[0.001,1] for i in range(len(ensemble_predictions))]),
                    verbose=True,random_state=2022, n_jobs=48,n_calls=100)


# In[26]:


best_weights=np.array(results['x'])/sum(results['x'])
print(exps)
print(best_weights)


# In[28]:


loss,ensemble_pred=ensemble_loss(best_weights,True)
print(loss)


# In[10]:


# output=preds.copy()
# output[['Ineffective','Adequate','Effective',]]=ensemble_pred
# output=output[['discourse_id','Ineffective','Adequate','Effective',]]
# output.to_csv("ensemble_oof.csv",index=False)

FOLDS=6

weights=best_weights

train_pl=pd.read_csv("../../input/train_pl.csv")
train_pl=train_pl.rename(columns={'id':"essay_id"})
train_pl['discourse_id']=[str(f) for f in train_pl['discourse_id']]
pl_discourse_ids=list(train_pl['discourse_id'])

train_pl[list(label_mapping.keys())]=0

for fold in range(FOLDS):
    train_pl[list(label_mapping.keys())]=0

    for exp,weight in zip(exps,weights):
        pl_labels=pd.read_csv(os.path.join(exp,f'pl_labels/{fold}.csv'))
        pl_labels['discourse_id']=[str(f) for f in pl_labels['discourse_id']]
        #pl_labels['label']=[vector for vector in pl_labels[list(label_mapping.keys())].values]

        train_pl_temp=pd.read_csv("../../input/train_pl.csv")
        train_pl_temp=train_pl_temp.rename(columns={'id':"essay_id"})
        train_pl_temp['discourse_id']=[str(f) for f in train_pl_temp['discourse_id']]
        #train_pl_temp[list(label_mapping.keys())]=0
        train_pl_temp=train_pl_temp.merge(pl_labels[['discourse_id','Ineffective', 'Adequate', 'Effective']],how='left',on='discourse_id')

        #na_indices=train_pl_temp['Ineffective']!=train_pl_temp['Ineffective']

        assert sum(train_pl_temp['Ineffective']!=train_pl_temp['Ineffective'])==sum(train_pl_temp['Adequate']!=train_pl_temp['Adequate'])==sum(train_pl_temp['Effective']!=train_pl_temp['Effective'])
        train_pl_temp=train_pl_temp.fillna(0.33333)

        train_pl[list(label_mapping.keys())]+=train_pl_temp[list(label_mapping.keys())]*weight

    train_pl[['discourse_id','Ineffective', 'Adequate', 'Effective']].to_csv(f"{folder}/{fold}.csv")

    print(train_pl[['discourse_id','Ineffective', 'Adequate', 'Effective']])
