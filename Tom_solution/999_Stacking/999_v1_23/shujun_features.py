import numpy as np
import pandas as pd
from tqdm import tqdm

def get_xgb_features(train_df, prob_sequences, use_prob_seq=True):
    '''
    prob_seq is the sequence of token probs from each discourse
    '''
    
    if use_prob_seq:
        # 10 features : instability_0,...,instability_3, begin_0,...,begin_2, end_0,...,end_2
        features2calculate=[f"instability_{i}" for i in range(4)]+\
        [f"begin_{i}" for i in range(3)]+\
        [f"end_{i}" for i in range(3)]
        calculated_features=[]
        for i,prob_seq in tqdm(enumerate(prob_sequences)):
            tmp=[]
            prob_seq=np.array(prob_seq)
            tmp.append(np.diff(prob_seq[:,:],0).mean(0)) # 3
            tmp.append([(np.diff(prob_seq[:,[1,2]].sum(1))**2).mean()]) # 1
            tmp.append(prob_seq[:5,:].mean(0)) # 3
            tmp.append(prob_seq[-5:,:].mean(0)) # 3
            calculated_features.append(np.concatenate(tmp))
        calculated_features=np.array(calculated_features)
        print('calculated_features.shape = ', calculated_features.shape)
        train_df[features2calculate]=calculated_features # 10 features
        train_df['len']=[len(s) for s in prob_sequences]

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
            try:
                row.append(value_count_hash[d][essay_id])
            except:
                row.append(0)
        discourse_cnts.append(row)

    discourse_cnts=pd.DataFrame(discourse_cnts,columns=['essay_id']+[f'{d}_count' for d in discourse_types])    

    train_df=train_df.merge(discourse_cnts,how='left',on='essay_id')

    return train_df