#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import log_loss
from tqdm import tqdm


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
# exps=["deberta_v3_large_pl_4th_tascj0","deberta_v2_xlarge_pl_4th_tascj0","deberta_large_pl_4th_tascj0"]
# exps+=["deberta_v3_large_pl_4th","deberta_v2_xlarge_pl_4th","deberta_xlarge_pl_4th_tascj0"]

exps=["deberta_v3_large_pl_5th_tascj0_corrected",'deberta_v2_xlarge_pl_5th_tascj0_corrected',
'deberta_xlarge_pl_5th_tascj0',
'deberta_v2_xlarge_pl_5th','deberta_v3_large_pl_5th']

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

    y_oof=np.load(f"{exp}/stacking_oofs.npy")

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


output=preds.copy()
output[['Ineffective','Adequate','Effective',]]=ensemble_pred
output=output[['discourse_id','Ineffective','Adequate','Effective',]]
output.to_csv("ensemble_oof.csv",index=False)


# In[29]:


# loss,ensemble_pred=ensemble_loss(np.ones(6),True)
# loss
