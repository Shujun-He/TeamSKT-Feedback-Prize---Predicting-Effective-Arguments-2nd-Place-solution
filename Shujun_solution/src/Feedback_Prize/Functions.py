from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import itertools
import ast
from sklearn.metrics import f1_score
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from tqdm import tqdm

# Return an array that maps character index to index of word in list of split() words
def split_mapping(unsplit):
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit),-1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx


class CustomCollate:
    def __init__(self,tokenizer,train=True,sliding_window=None):
        self.tokenizer=tokenizer
        self.train=train
        self.sliding_window=sliding_window

    def __call__(self,data):
        """
        need to collate: input_ids, attention_mask, labels
        input_ids is padded with 1, attention_mask 0, labels -100

        """


        bs=len(data)
        # print(data[0])
        # exit()
        lengths=[]
        for i in range(bs):
            lengths.append(len(data[i]['input_ids']))
        max_len=max(lengths)
        if self.sliding_window is not None and max_len > self.sliding_window:
            max_len= int((np.floor(max_len/self.sliding_window-1e-6)+1)*self.sliding_window)
        #max_len=1024
        input_ids, attention_mask, labels, BIO_labels, discourse_labels=[],[],[],[],[]
        sequence_ids=[]
        gather_indices=[]
        wids=[]
        discourse_ids=[]
        discourse_type_ids=[]
        for i in range(bs):
            input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(0,max_len-lengths[i]),value=self.tokenizer.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(0,max_len-lengths[i]),value=0))
            labels.append(data[i]['labels'])
            sequence_ids.append(torch.nn.functional.pad(data[i]['sequence_ids'],(0,max_len-lengths[i]),value=-1))
            gather_indices.append(torch.nn.functional.pad(data[i]['gather_indices'],(0,max_len-lengths[i]),value=-1))
            discourse_type_ids.append(torch.nn.functional.pad(data[i]['discourse_type_ids'],(0,max_len-lengths[i]),value=0))
            discourse_ids=discourse_ids+data[i]['discourse_ids']
            #wids.append(torch.nn.functional.pad(data[i]['wids'],(0,max_len-lengths[i]),value=-1))
        input_ids=torch.stack(input_ids)
        attention_mask=torch.stack(attention_mask)
        labels=torch.cat(labels)
        sequence_ids=torch.stack(sequence_ids)
        gather_indices=torch.stack(gather_indices)
        discourse_type_ids=torch.stack(discourse_type_ids)
        #wids=torch.stack(wids)

        gather_indices_by_row=gather_indices
        # for i in range(1,bs):
        #     gather_indices[i][gather_indices[i]!=-1]=gather_indices[i][gather_indices[i]!=-1]+gather_indices[i-1].max()+1
        #     print(gather_indices[i-1].max())
        # exit()

        # print(gather_indices[1][:100])
        # print(gather_indices[2][:100])
        # print(torch.unique(gather_indices))
        # print(labels)
        # print(len(torch.unique(gather_indices)))
        # print(len(labels))
        #
        #
        # exit()

        #offsets=[encoding["offset_mapping"] for encoding in data]
        offsets=[]
        # print(len(offsets[0]))
        # exit()

        return {"input_ids":input_ids,"attention_mask":attention_mask,
        "labels":labels,"sequence_ids":sequence_ids,"wids":wids,"offsets":offsets,
        "sample_id":np.arange(len(input_ids)),"gather_indices":gather_indices,"discourse_ids":discourse_ids,
        "discourse_type_ids":discourse_type_ids,"gather_indices_by_row":gather_indices_by_row}



def custom_collate_train(data):
    """
    need to collate: input_ids, attention_mask, labels
    input_ids is padded with 1, attention_mask 0, labels -100

    """

    bs=len(data)
    lengths=[]
    for i in range(bs):
        lengths.append(len(data[i]['input_ids']))
        # print(data[i]['input_ids'].shape)
        # print(data[i]['attention_mask'].shape)
        # print(data[i]['labels'].shape)
    max_len=max(lengths)

    #always pad the right side
    input_ids, attention_mask, labels=[],[],[]
    #if np.random.uniform()>0.5:
    for i in range(bs):
        input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(0,max_len-lengths[i]),value=tokenizer.pad_token_id))
        attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(0,max_len-lengths[i]),value=0))
        labels.append(torch.nn.functional.pad(data[i]['labels'],(0,max_len-lengths[i]),value=-100))
    # else:
    #     for i in range(bs):
    #         input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(max_len-lengths[i],0),value=1))
    #         attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(max_len-lengths[i],0),value=0))
    #         labels.append(torch.nn.functional.pad(data[i]['labels'],(max_len-lengths[i],0),value=-100))

    input_ids=torch.stack(input_ids)
    attention_mask=torch.stack(attention_mask)
    labels=torch.stack(labels)
    #exit()

    return {"input_ids":input_ids,"attention_mask":attention_mask,"labels":labels}


def iter_split(data,labels,fold,nfolds=5,seed=2020):
    splits = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    splits = list(splits.split(data,labels))
    # splits = np.zeros(len(data)).astype(np.int)
    # for i in range(nfolds): splits[splits[i][1]] = i
    # indices=np.arange(len(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    return train_indices, val_indices


# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

def threshold(df):
    map_clip = {'Lead':9, 'Position':5, 'Evidence':14, 'Claim':3, 'Concluding Statement':11,
                 'Counterclaim':6, 'Rebuttal':4}
    df = df.copy()
    for key, value in map_clip.items():
    # if df.loc[df['class']==key,'len'] < value
        index = df.loc[df['class']==key].query(f'len<{value}').index
        df.drop(index, inplace = True)
    return df

from tqdm import tqdm
def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

def link_evidence(oof):
  if not len(oof):
    return oof

  def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

  thresh = 1
  idu = oof['id'].unique()
  eoof = oof[oof['class'] == "Evidence"]
  neoof = oof[oof['class'] != "Evidence"]
  eoof.index = eoof[['id', 'class']]
  for thresh2 in range(26, 27, 1):
    retval = []
    for idv in tqdm(idu, desc='link_evidence', leave=False):
      for c in ['Evidence']:
        q = eoof[(eoof['id'] == idv)]
        if len(q) == 0:
          continue
        pst = []
        for r in q.itertuples():
          pst = [*pst, -1,  *[int(x) for x in r.predictionstring.split()]]
        start = 1
        end = 1
        for i in range(2, len(pst)):
          cur = pst[i]
          end = i
          if  ((cur == -1) and ((pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
            retval.append((idv, c, jn(pst, start, end)))
            start = i + 1
        v = (idv, c, jn(pst, start, end + 1))
        retval.append(v)
    roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
    roof = roof.merge(neoof, how='outer')
    return roof


def get_char_probs(texts, features_texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, features_text, prediction) in enumerate(zip(texts, features_texts, predictions)):
        encoded = tokenizer(features_text, text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        text_indices=np.array(encoded.sequence_ids())==1


        #print(text)
        for idx, (offset_mapping, pred) in enumerate(zip(np.array(encoded['offset_mapping'])[text_indices], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred



            #if text[start:start+1]=='\n' or '\t' or '\r':
            # print(text[start:end])
            # exit()
            #print(text[start:start+1])
            # if text[start:start+1]=='\n':
            #     print(text[start:end])
            #     results[i][start:start+1] = 0

    #exit()

    return results

def get_char_probs_merge_words(texts, features_texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, features_text, prediction) in enumerate(zip(texts, features_texts, predictions)):
        encoded = tokenizer(features_text, text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        text_indices=np.array(encoded.sequence_ids())==1
        word_ids = encoded.word_ids()
        word_ids=np.array(word_ids)
        word_ids[word_ids==None]=-1
        word_ids=word_ids[np.array(encoded.sequence_ids())==1]
        # assert len(word_ids)==len(prediction)
        # print(len(word_ids))
        # print(len(prediction))
        # print(word_ids)
        # print(prediction)
        word_ids=np.array(word_ids)
        prediction=np.array(prediction)
        # print(word_ids)
        #print(prediction[0])
        # print(np.max(word_ids))
        # exit()
        # for j in range(np.max(word_ids)):
        #     prediction[word_ids==j]=np.mean(prediction[word_ids==j])
            # prediction[word_ids==i]=prediction[word_ids==i]
            # pass
            # prediction[i]=prediction[i]
        # print(prediction[0])
        # print("###")
        #exit()
        # print(word_ids)
        # exit()

        for idx, (offset_mapping, pred) in enumerate(zip(np.array(encoded['offset_mapping'])[text_indices], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results

# def get_results(char_probs, texts, th=0.5):
#     results = []
#     for char_prob,text in zip(char_probs,texts):
#         result = np.where(char_prob >= th)[0] + 1
#         result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
#
#         new_result=[]
#         #for r,text in zip(result,texts):
#         for r in result:
#             start,end=min(r),max(r)
#             if text[start:start+1]=='\n' or text[start:start+1]=='\t' or text[start:start+1]=='\r':
#                 new_result.append(f"{min(r)+1} {max(r)}")
#             elif text[start-1:start]!='\n' and text[start-1:start]!='\t' and text[start-1:start]!='\r' and  text[start-1:start]!=' ':
#                 new_result.append(f"{min(r)-1} {max(r)}")
#             # elif text[start-1:start]=='2':
#             #     new_result.append(f"{min(r)-1} {max(r)}")
#             else:
#                 new_result.append(f"{min(r)} {max(r)}")
#         result=new_result
#         #result = [f"{min(r)} {max(r)}" for r in result]
#         result = ";".join(result)
#         results.append(result)
#     return results


def get_results(char_probs, texts, th=0.5):
    results = []
    for char_prob,text in zip(char_probs,texts):
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]

        new_result=[]
        #for r,text in zip(result,texts):
        for r in result:
            start,end=min(r),max(r)
            if text[start:start+1]=='\n' or text[start:start+1]=='\t' or text[start:start+1]=='\r':
                #new_result.append(f"{min(r)+1} {max(r)}")
                start=start+1
            elif text[start-1:start]!='\n' and text[start-1:start]!='\t' and text[start-1:start]!='\r' and  text[start-1:start]!=' ':
                #new_result.append(f"{min(r)-1} {max(r)}")
                start=start-1
            # start=max(0,start)
            # start=min(start,len(text))

            # if start>=end:
            #     print("shit")
            # if start<0:
            #     print("shit")
            # if start>len(text):
            #     print("shit")
            if start<end:
                new_result.append(f"{start} {end}")
#             else:
#                 new_result.append(f"{min(r)} {max(r)}")
        result=new_result
        #result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results

# def get_results(char_probs, th=0.5):
#     results = []
#     for char_prob in char_probs:
#         result = np.where(char_prob >= th)[0]
#         result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
#         result = [f"{min(r)} {max(r)+1}" for r in result]
#         result = ";".join(result)
#         results.append(result)
#     return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions

def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths

# From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)

def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score


def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
#     txt = re.sub(r'\s+', ' ', txt)
    return txt


def load_and_prepare_data(df,patient_notes,features):
    # patient_notes = pd.read_csv(root + "patient_notes.csv")
    # features = pd.read_csv(root + "features.csv")
    #df = pd.read_csv(root + "test.csv")

    df = df.merge(features, how="left", on=["case_num", "feature_num"])
    df = df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

    # print(df.columns)
    # exit()

    df['pn_history'] = df['pn_history'].apply(lambda x: x.strip())
    df['feature_text'] = df['feature_text'].apply(process_feature_text)

    df['feature_text'] = df['feature_text'].apply(clean_spaces)
    df['clean_text'] = df['pn_history'].apply(clean_spaces)

    df['target'] = ""
    return df
