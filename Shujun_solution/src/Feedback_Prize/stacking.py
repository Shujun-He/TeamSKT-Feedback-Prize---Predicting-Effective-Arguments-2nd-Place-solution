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


train_kaggle=pd.read_csv("../../../input/feedback-prize-effectiveness/train.csv")
train_kaggle


# In[3]:


save_path="./gbm_models"
oof_path="."
NFOLDS=6
train_df=[]
prob_sequences=[]
for fold in range(NFOLDS):
    df=pd.read_csv(f"{oof_path}/oofs/{fold}.csv")
    df['kfold']=fold
    train_df.append(df)
    with open(f"{oof_path}/oofs/{fold}.p",'rb') as f:
        prob_sequences+=pickle.load(f)


train_df=pd.concat(train_df)
if 'discourse_type' not in train_df.columns:
    train_df=train_df.merge(train_kaggle[['discourse_id','discourse_type','essay_id']],how='left',on='discourse_id')
else:
    train_df=train_df.merge(train_kaggle[['discourse_id','essay_id']],how='left',on='discourse_id')
train_df['discourse_type']=train_df['discourse_type'].astype("category")
train_df


# In[4]:


def sorted_quantile(array, q):
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction

from scipy.stats import entropy


# In[5]:


# #make features
# n_quan=7

# train_df[[f"instability_{i}" for i in range(4)]]=0.0
# train_df[[f"begin_{i}" for i in range(3)]]=0.0
# train_df[[f"end_{i}" for i in range(3)]]=0.0
# train_df[[f"quan_{i}" for i in range(n_quan*3)]]=0.0
# train_df["entropy"]=0.0
# for i,prob_seq in tqdm(enumerate(prob_sequences)):
#     #quants = np.linspace(0,1,n_quan)
#     prob_seq=np.array(prob_seq)
#     instability = []
#     #all_quants=[]
#     for j in range(3):
#         #if len(s) > 1:
#         train_df[f"instability_{j}"].iloc[i]=(np.diff(prob_seq[:,j])**2).mean()
#         train_df[f"begin_{j}"].iloc[i]=(prob_seq[:5,j]).mean()
#         train_df[f"end_{j}"].iloc[i]=(prob_seq[-5:,j]).mean()

#         #all_quants+=list(sorted_quantile(prob_seq[:,j], quants))
#     train_df[f"instability_3"].iloc[i]=(np.diff(prob_seq[:,[1,2]].sum(1))**2).mean()
#     train_df["entropy"].iloc[i]=np.mean([entropy(s) for s in prob_seq])
#     #train_df[[f"quan_{i}" for i in range(n_quan*3)]].iloc[i]=all_quants
#     #train_df[[f"instability_{j}" for j in range(3)]].iloc[i]=instability
#     #.iloc[i]=instability[0]
# train_df['len']=[len(s) for s in prob_sequences]


# In[6]:


features=["Ineffective","Adequate","Effective",
          "instability_0","instability_1","instability_2","instability_3",
          "len","discourse_type"]
features+=[f"begin_{i}" for i in range(3)]
features+=[f"end_{i}" for i in range(3)]
#features=["Ineffective","Adequate","Effective",]
#features+=[f"quan_{i}" for i in range(n_quan*3)]
target='label'
features


# In[7]:


#%%time

features2calculate=[f"instability_{i}" for i in range(4)]+[f"begin_{i}" for i in range(3)]+[f"end_{i}" for i in range(3)]#+\
#["entropy"]

calculated_features=[]
for i,prob_seq in tqdm(enumerate(prob_sequences)):

    tmp=[]
    #quants = np.linspace(0,1,n_quan)
    prob_seq=np.array(prob_seq)
    instability = []
    #all_quants=[]
    tmp.append(np.diff(prob_seq[:,:],0).mean(0))
    tmp.append([(np.diff(prob_seq[:,[1,2]].sum(1))**2).mean()])

    tmp.append(prob_seq[:5,:].mean(0))
    tmp.append(prob_seq[-5:,:].mean(0))

    #tmp.append(np.mean([entropy(s) for s in prob_seq]))


    calculated_features.append(np.concatenate(tmp))

#     for j in range(3):
#         #if len(s) > 1:
#         train_df[f"instability_{j}"].iloc[i]=(np.diff(prob_seq[:,j])**2).mean()
#         train_df[f"begin_{j}"].iloc[i]=(prob_seq[:5,j]).mean()
#         train_df[f"end_{j}"].iloc[i]=(prob_seq[-5:,j]).mean()

#         #all_quants+=list(sorted_quantile(prob_seq[:,j], quants))
#     train_df[f"instability_3"].iloc[i]=(np.diff(prob_seq[:,[1,2]].sum(1))**2).mean()
#     train_df["entropy"].iloc[i]=np.mean([entropy(s) for s in prob_seq])
    #train_df[[f"quan_{i}" for i in range(n_quan*3)]].iloc[i]=all_quants
    #train_df[[f"instability_{j}" for j in range(3)]].iloc[i]=instability

train_df[features2calculate]=calculated_features
train_df['len']=[len(s) for s in prob_sequences]

calculated_features=np.array(calculated_features)
calculated_features.shape

p_features=[]
n_features=[]
neighbor_features=['Ineffective','Adequate','Effective','discourse_type']
neighbor_features_values=train_df[neighbor_features].values
for i in tqdm(range(len(train_df))):

    if i>1 and train_df['essay_id'].iloc[i]==train_df['essay_id'].iloc[i-1]:
        p_features.append(neighbor_features_values[i-1])
    else:
        p_features.append(neighbor_features_values[i])

    if i<(len(train_df)-1) and train_df['essay_id'].iloc[i]==train_df['essay_id'].iloc[i+1]:
        n_features.append(neighbor_features_values[i+1])
    else:
        n_features.append(neighbor_features_values[i])

train_df[[f+"_previous" for f in neighbor_features]]=p_features
train_df[[f+"_next" for f in neighbor_features]]=n_features

train_df['mean_Ineffective']=train_df.groupby("essay_id")["Ineffective"].transform("mean")
train_df['mean_Adequate']=train_df.groupby("essay_id")["Adequate"].transform("mean")
train_df['mean_Effective']=train_df.groupby("essay_id")["Effective"].transform("mean")

train_df['std_Ineffective']=train_df.groupby("essay_id")["Ineffective"].transform("std")
train_df['std_Adequate']=train_df.groupby("essay_id")["Adequate"].transform("std")
train_df['std_Effective']=train_df.groupby("essay_id")["Effective"].transform("std")

train_df['discourse_count']=train_df.groupby("essay_id")['discourse_type'].transform("count")

cnts=train_df.groupby('essay_id')['discourse_type'].apply(lambda x: x.value_counts())

#new_df=[]
discourse_types=['Claim','Evidence','Concluding Statement','Lead','Position','Counterclaim','Rebuttal']
value_count_hash={}
for t in discourse_types:
    value_count_hash[t]={}
for key in cnts.keys():
    value_count_hash[key[1]][key[0]]=cnts[key]

discourse_cnts=[]
for essay_id in train_df['essay_id'].unique():
    row=[essay_id]
    for d in discourse_types:
        row.append(value_count_hash[d][essay_id])
    discourse_cnts.append(row)

discourse_cnts=pd.DataFrame(discourse_cnts,columns=['essay_id']+[f'{d}_count' for d in discourse_types])
discourse_cnts

train_df=train_df.merge(discourse_cnts,how='left',on='essay_id')
train_df

train_df


# In[8]:


len(prob_sequences)


# In[9]:


features=["Ineffective","Adequate","Effective",
          "instability_0","instability_1","instability_2","instability_3",
          "len","discourse_type"]
features+=[f"begin_{i}" for i in range(3)]
features+=[f"end_{i}" for i in range(3)]

features=features+[f+"_previous" for f in neighbor_features]+[f+"_next" for f in neighbor_features]+['mean_Ineffective','mean_Adequate','mean_Effective']+['std_Ineffective','std_Adequate','std_Effective']+['discourse_count']+[f'{d}_count' for d in discourse_types]


#train_df['discourse_type_previous']=train_df['discourse_type_previous'].astype("category")
#train_df['discourse_type_next']=train_df['discourse_type_next'].astype("category")
features


# In[10]:


for f in features:
    if f not in ['discourse_type_previous','discourse_type_next','discourse_type']:
        train_df[f]= train_df[f].astype('float')
    else:
        train_df[f]= train_df[f].astype('category')


# In[11]:


import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
import os

os.system(f'mkdir {save_path}')
use_gpu=True

if use_gpu:
    xgb_param = {'objective': 'multi:softprob',
             'eval_metric': "mlogloss",
             'learning_rate': 0.05,
             'max_depth': 2,
             "min_child_weight": 10,
             "colsample_bynode": 0.6,
             "subsample": 1,
             "num_class":3,
             "tree_method": 'gpu_hist', "gpu_id": 1
        }
else:

    xgb_param = {'objective': 'multi:softprob',
             'eval_metric': "mlogloss",
             'learning_rate': 0.05,
             'max_depth': 2,
             "min_child_weight": 10,
             "colsample_bynode": 0.6,
             "subsample": 1,
             "num_class":3,
        }

TRAIN_XGB_N_FOLDS=NFOLDS

y_oof = np.zeros((train_df.shape[0],3))
res = dict()
best_th = dict()

lvl1_stacking_df = []

#for dtype in discourse_types:
#     if dtype == "Evidence":
#         lgb_param["boosting"] = "gbdt"
#         lgb_param["learning_rate"] = 0.05
#     else:
#         lgb_param["boosting"] = "dart"
#         lgb_param["learning_rate"] = 0.1
# lgb_param["boosting"] = "gbdt"
# lgb_param["learning_rate"] = 0.01

#features = features_dict[dtype]


all_indices = np.arange(len(train_df))
#discourse_df = train_df.reset_index(drop=True)

#tm = discourse_df["target"].mean()

#xgb_param["scale_pos_weight"] = (1 - tm)/tm

best_its = []

#print(dtype, len(features))
for f in range(TRAIN_XGB_N_FOLDS):
    print(f"training for fold {f}")
    val_ind = all_indices[np.where(train_df["kfold"] == f)[0]]
    train_indices=train_df["kfold"] != f
    val_indices=train_df["kfold"] == f
    train, val_df = train_df[train_indices], train_df[val_indices]
    #one_hot_labels
    d_train = xgb.DMatrix(train[features], train[target],enable_categorical=True)
    d_val = xgb.DMatrix(val_df[features], val_df[target],enable_categorical=True)

    model = xgb.train(xgb_param, d_train, evals=[(d_val, "val")], num_boost_round=2000, verbose_eval=200,
                      early_stopping_rounds=50)
    model.save_model(f'{save_path}/xgb_{f}.json')
    pickle.dump(model, open(f'{save_path}/xgb_{f}.p', "wb+"))
    y_oof[val_ind] = model.predict(d_val)
    best_its.append(model.best_iteration)


log_loss(train_df[target],y_oof)


# In[12]:


print("score with stacking")
print(log_loss(train_df[target],y_oof))


# In[13]:


print("score without stacking")

print(log_loss(train_df[target],train_df[["Ineffective","Adequate","Effective"]]))


# In[14]:


y_oof.shape


# In[15]:


train_df.shape


# In[16]:


np.save('stacking_oofs.npy',y_oof)
np.save('nn_oofs.npy',train_df[["Ineffective","Adequate","Effective"]].values)
