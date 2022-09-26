import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Functions import *
from scipy import stats
import ast
import random
#import nlpaug.augmenter.word as naw
import os


def get_substring_span(text, substring, min_length=10, fraction=0.999):
    """
    Returns substring's span from the given text with the certain precision.
    """

    position = text.find(substring)
    substring_length = len(substring)
    if position == -1:
        half_length = int(substring_length * fraction)
        half_substring = substring[:half_length]
        half_substring_length = len(half_substring)
        if half_substring_length < min_length:
            return [-1, 0]
        else:
            return get_substring_span(text=text,
                                    substring=half_substring,
                                    min_length=min_length,
                                    fraction=fraction)

    span = [position, position+substring_length]
    return span

discourse_mapping={'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3, 'Counterclaim': 4, 'Rebuttal': 5, 'Concluding Statement': 6}

class FeedbackDataset(Dataset):
    def __init__(self, tokenizer, df, full_texts, train, aug=False, loss_type="BCELoss", max_len=512):
        self.tokenizer = tokenizer
        self.texts = df['discourse_text'].values
        self.labels = df['label'].values
        self.discourse_type=df['discourse_type'].values
        self.essay_ids=df['essay_id'].values
        self.full_texts=full_texts
        self.max_len = max_len
        self.aug = aug
        #self.nlp_aug=naw.SynonymAug()
        self.train=train
        self.df=df
        self.encodings=[]
        self.labels=[]
        self.gather_indices=[]
        self.discourse_ids=[]
        self.discourse_type_ids=[]
        total=0

        self.essay_ids=list(df['essay_id'].unique())

        # for key in tqdm(df['essay_id'].unique()):
        #     discourses=df[df['essay_id']==key]
        #     text=full_texts[key]
        #     reference_text=text[:]
        #
        #     for discourse_text,label,id,type in zip(discourses['discourse_text'],discourses['label'],discourses['discourse_id'],discourses['discourse_type']):
        #         span=get_substring_span(reference_text, discourse_text.strip())
        #         text=text[:span[0]]+f"({type} start)"+discourse_text.strip()+f"({type} end)"+text[span[1]:]
        #         reference_text=reference_text[:span[0]]+f"({type} start)"+"*"*(span[1]-span[0])+f"({type} end)"+reference_text[span[1]:]
        #
        #
        #         #reference_text[:span[0]]+"*"*(span[1]-span[0])+text[span[1]:]
        #
        #     encoding = self.tokenizer(text,
        #                            add_special_tokens=True,
        #                            max_length=self.max_len,
        #                            padding=False,
        #                            return_offsets_mapping=True,
        #                            truncation=True)
        #     gather_indices=np.ones(len(encoding['input_ids']))*-1
        #     discourse_type_ids=np.zeros(len(encoding['input_ids']))
        #     cnt=0
        #     sample_labels=[]
        #     discourse_ids=[]
        #
        #
        #
        #     for discourse_text,label,id,type in zip(discourses['discourse_text'],discourses['label'],discourses['discourse_id'],discourses['discourse_type']):
        #         span=get_substring_span(text, discourse_text.strip())
        #         n_tokens=0
        #         # print(encoding['offset_mapping'])
        #         # exit()
        #         for i in range(len(gather_indices)):
        #             if encoding['offset_mapping'][i]!=(0,0) and encoding['offset_mapping'][i][0]>=span[0] and encoding['offset_mapping'][i][1]<=span[1]:
        #                 gather_indices[i]=cnt
        #                 discourse_type_ids[i]=discourse_mapping[type]
        #                 n_tokens+=1
        #         text=text[:span[0]]+"*"*(span[1]-span[0])+text[span[1]:]
        #         # if (gather_indices==3).sum()==0:
        #         #     print(gather_indices)
        #         if (gather_indices==cnt).sum()>0:
        #             sample_labels.append(label)
        #             discourse_ids.append(id)
        #             cnt+=1
        #
        #     self.encodings.append(encoding)
        #     self.labels.append(sample_labels)
        #     self.gather_indices.append(gather_indices)
        #     self.discourse_ids.append(discourse_ids)
        #     self.discourse_type_ids.append(discourse_type_ids)
        #     total+=len(sample_labels)
            # print(gather_indices)
            # print(sample_labels)
            #
            # print(key)
            # print(discourses)
        #exit()
        #self.discourse_ids
        # print(total/len(df))
        # exit()


        # self.anchors = df['anchor'].values
        # self.targets = df['target'].values
        # self.contexts = df['context'].values
        # if loss_type=='BCELoss':
        #     self.labels = df['score'].values
        # elif loss_type=='CrossEntropyLoss':
        #     self.labels = df['score_map'].values
        # elif loss_type=='OrdinalLoss':
        #     self.labels=[]
        #     for label in df['score_map'].values:
        #         temp=np.zeros(4)
        #         temp[:label]=1
        #         self.labels.append(temp)
            #self.labels = df['score_map'].values


        #self.level=level
    def get_essay_data(self,idx):
        key=self.essay_ids[idx]
        discourses=self.df[self.df['essay_id']==key]
        text=self.full_texts[key]
        reference_text=text[:]

        for discourse_text,label,id,type in zip(discourses['discourse_text'],discourses['label'],discourses['discourse_id'],discourses['discourse_type']):
            span=get_substring_span(reference_text, discourse_text.strip())
            text=text[:span[0]]+f"({type} start)"+discourse_text.strip()+f"({type} end)"+text[span[1]:]
            reference_text=reference_text[:span[0]]+f"({type} start)"+"*"*(span[1]-span[0])+f"({type} end)"+reference_text[span[1]:]


            #reference_text[:span[0]]+"*"*(span[1]-span[0])+text[span[1]:]

        encoding = self.tokenizer(text,
                               add_special_tokens=True,
                               max_length=self.max_len,
                               padding=False,
                               return_offsets_mapping=True,
                               truncation=True)
        gather_indices=np.ones(len(encoding['input_ids']))*-1
        discourse_type_ids=np.zeros(len(encoding['input_ids']))
        cnt=0
        sample_labels=[]
        discourse_ids=[]



        for discourse_text,label,id,type in zip(discourses['discourse_text'],discourses['label'],discourses['discourse_id'],discourses['discourse_type']):
            span=get_substring_span(text, discourse_text.strip())
            n_tokens=0
            # print(encoding['offset_mapping'])
            # exit()
            for i in range(len(gather_indices)):
                if encoding['offset_mapping'][i]!=(0,0) and encoding['offset_mapping'][i][0]>=span[0] and encoding['offset_mapping'][i][1]<=span[1]:
                    gather_indices[i]=cnt
                    discourse_type_ids[i]=discourse_mapping[type]
                    n_tokens+=1
            text=text[:span[0]]+"*"*(span[1]-span[0])+text[span[1]:]
            # if (gather_indices==3).sum()==0:
            #     print(gather_indices)
            if (gather_indices==cnt).sum()>0:
                sample_labels.append(label)
                discourse_ids.append(id)
                cnt+=1

        # self.encodings.append(encoding)
        # self.labels.append(sample_labels)
        # self.gather_indices.append(gather_indices)
        # self.discourse_ids.append(discourse_ids)
        # self.discourse_type_ids.append(discourse_type_ids)
        return encoding,sample_labels,gather_indices,discourse_ids,discourse_type_ids


    def __len__(self):
        return len(self.essay_ids)


            # for text in

    def __getitem__(self, idx):
        encoding,sample_labels,gather_indices,discourse_ids,discourse_type_ids=self.get_essay_data(idx)

        #encoding=self.encodings[idx]
        encoding['wids']=np.array(encoding.word_ids())
        encoding['wids'][encoding['wids']==None]=-1
        encoding['wids']=encoding['wids'].astype('int')
        #encoding.sequence_ids()
        label = sample_labels
        sequence_ids=np.array(encoding.sequence_ids())
        sequence_ids[sequence_ids==None]=-1
        # print(sequence_ids)
        # exit()

        data={k:torch.tensor(v, dtype=torch.long) for k,v in encoding.items()}
        data['labels']=torch.tensor(label, dtype=torch.float)
        data['sequence_ids']=torch.tensor(sequence_ids.astype("int"))
        data['gather_indices']=torch.tensor(gather_indices)
        data['discourse_ids']=discourse_ids
        data['discourse_type_ids']=torch.tensor(discourse_type_ids)
        return data
