import pandas as pd
import os
import numpy as np


#exps=["deberta_1280_linear_anneal"]+[f"deberta_1280_linear_anneal_{i}" for i in range(33)]

exps=['deberta_v3_large_tascj0','deberta_v2_xlarge_tascj0','deberta_v3_large_pl_tascj0','deberta_v2_xlarge_pl_tascj0','deberta_v3_large_pl_2nd_tascj0','deberta_v2_xlarge_pl_2nd_tascj0','deberta_v3_large_pl_3rd_tascj0','deberta_v2_xlarge_pl_3rd_tascj0','deberta_v3_large_pl_3rd_tascj0_LSTM','deberta_v2_xlarge_pl_3rd_tascj0_LSTM','deberta_v2_xlarge_pl_4th_tascj0','deberta_v3_large_pl_4th_tascj0','deberta_large_pl_4th_tascj0','deberta_v3_large_pl_4th_tascj0_more_epochs','deberta_large_pl_4th_tascj0_more_epochs','longformer_large_pl_4th_tascj0','deberta_xlarge_pl_4th_tascj0','deberta_v3_large_pl_4th','deberta_v2_xlarge_pl_4th',"deberta_v3_large_pl_5th_tascj0",'deberta_v2_xlarge_pl_5th_tascj0',"deberta_v3_large_pl_5th_tascj0_corrected",'deberta_v2_xlarge_pl_5th_tascj0_corrected','deberta_large_pl_5th_tascj0','deberta_xlarge_pl_5th_tascj0','deberta_v2_xlarge_pl_5th','deberta_v3_large_pl_5th']


cvs=[]
for i,exp in enumerate(exps):
    cv=[]
    for j in range(6):
    #for j in [0,1,3,4,5]:
        df=pd.read_csv(f"{exp}/logs/fold{j}.csv")
        #best_epoch=df['epoch'].iloc[df['val_loss'].argmin()]
        #print(f"for original best epoch: {best_epoch} with score: {df['val_loss'].iloc[best_epoch]} for fold {i}")
        cv.append(df['val_loss'].iloc[df['val_loss'].argmin()])
        #os.system(f'cp models/fold{i}_epoch{best_epoch}.pt best_weights/fold{i}.pt')
        #df=pd.read_csv(f"logs/fold{i}_rnn.csv")
        #best_epoch=df['epoch'].iloc[df['val_score'].argmax()]
        #print(f"for RNN :    best epoch: {best_epoch} with score: {df['val_score'].iloc[best_epoch]} for fold {i}")

        #df=pd.read_csv(f"logs/fold{i}_cnn.csv")
        #best_epoch=df['epoch'].iloc[df['val_score'].argmax()]
        #print(f"for CNN :    best epoch: {best_epoch} with score: {df['val_score'].iloc[best_epoch]} for fold {i}")
    #for i in range(8):
        #print(" ")
    print(i)
    print(f"{exp} avg val score {np.mean(cv)}")
    cvs.append(np.mean(cv))

#print(np.argmin(cvs))
