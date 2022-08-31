import pandas as pd
from tqdm import tqdm

relation_mapper = {
    'Lead': ['Position',''],
    'Position': ['Lead',''],
    'Claim': ['Position',''],
    'Counterclaim': ['Position',''],
    'Rebuttal': ['Counterclaim',''],
    'Evidence': ['Claim',''],
    'Concluding Statement': ['Claim', 'Evidence']
}


def preprocessing(df, num_neighbor=3):
    unique_ids = sorted(df['essay_id'].unique())
    essay_df = []
    for essay_id in tqdm(unique_ids):
        tmp_essay_df = df[df['essay_id']==essay_id].reset_index(drop=True)
        for i in range(len(tmp_essay_df)):
            text = ''
            tmp_df = tmp_essay_df.iloc[max(0,i-num_neighbor):min(i+num_neighbor,len(tmp_essay_df))].reset_index(drop=True)
            for j in range(len(tmp_df)):
                sample = tmp_df.iloc[j]
                text += f'[{sample["discourse_type"].upper()}]{sample["discourse_text"]}'
            tmp_essay_df.loc[i,'neighbor_text'] = text
        essay_df.append(tmp_essay_df)
    essay_df = pd.concat(essay_df).reset_index(drop=True)
    print('essay_df.shape = ', essay_df.shape)
    df = df.merge(essay_df[['discourse_id','neighbor_text']], on='discourse_id', how='left')
    return df