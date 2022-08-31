# kaggle-feedback-prize-effectiveness, Tom's part of the 2nd place solution
This is the training code of Tom's part of the 2nd place solution for the Kaggle competition, Feedback Prize - Predicting Effective Arguments.

Team's solution summary : https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347359  
Inference code for Tom's part : https://www.kaggle.com/code/tikutiku/feedback2-stacking

## solution overview for Tom's part
![part_of_solution_Tom](https://user-images.githubusercontent.com/10670302/186918739-9508425b-221f-4430-b4d2-2e7d0d0e579b.png)


## HARDWARE
Ubuntu 18.04 LTS (2TB boot disk)  
Core i7-10700K  
64GB memory  
1 x NVIDIA GeForce RTX3090  

## SOFTWARE
Python 3.7.3  
CUDA 11.0  
cuddn 8.4.0  
nvidia drivers v.460  


## Usage
0. download data from Kaggle  
this competition's data : https://www.kaggle.com/competitions/feedback-prize-effectiveness/data  
2021 competition's data : https://www.kaggle.com/competitions/feedback-prize-2021/data  
tascj0's pretrained weights : https://www.kaggle.com/datasets/tascj0/feedback-checkpoints  

put them as follows, respectively 
```
input/feedback-prize-effectiveness
input/feedback-prize-2021
input/tascj
```

run input/tascj/converter.ipynb for deberta_large_fold0.pth, deberta_xlarge_fold0.pth


1. data split for kaggle train data  
```
cd 00_EDA/00_v2_07/
run feedback2-playground.ipynb
```


2. data split for 2021 competition's data (for pretraining)
```
cd 00_EDA/00_v2_05/
run feedback1-playground.ipynb
```


3. data preparation for 2021 competition's data (for training with pseudo-label)
```
cd 00_EDA/00_v2_11/
run feedback2-playground.ipynb
```


4. data split for 2021 competition's data (for training with pseudo-label)
```
cd 00_EDA/00_v1_13/
run feedback2-playground.ipynb
```


5. pretrain span detector with 2021 competition's data
```
cd 05_Detection/exp
run 05_v2_09.ipynb
run 05_v2_11.ipynb
```


6. pretrain MLM with 2021 competition's data
```
cd 21_MLM2/exp
run 21_v1_10.ipynb
run 21_v2_14.ipynb
```


7. train base level models with span detector pretrained weights
```
cd 18_BaselineSW/exp
run 18_v2_01.ipynb
run 18_v2_03.ipynb
run 18_v1_05.ipynb
run 18_v2_22.ipynb
```


8. train base level models with huggingface pretrained weights
```
cd 20_WoSpanDet/exp
run 20_v1_01.ipynb
```


9. train base level models with MLM pretrained weights
```
cd 22_BaselineMLM/exp
run 22_v1_01.ipynb
run 22_v2_01.ipynb
```


10. generate 1st round pseudo-label of each model
```
cd 18_BaselineSW/exp
run 18_v2_01-pseudo.ipynb
run 18_v2_03-pseudo.ipynb
run 18_v1_05-pseudo.ipynb
run 18_v2_22-pseudo.ipynb
```

```
cd 20_WoSpanDet/exp
run 20_v1_01-pseudo.ipynb
```

```
cd 22_BaselineMLM/exp
run 22_v1_01-pseudo.ipynb
run 22_v2_01-pseudo.ipynb
```


11. train with 1st round pseudo-label, then finetune on gt label of this competition
```
cd 29_Pseudo4
run code/ensemble_to_create_pseudo_label.ipynb 
run exp/29_v2_01.ipynb
run exp/29_v2_02.ipynb
run exp/29_vl_01.ipynb
```


12. generate 2nd round pseudo-label of each model
```
cd 29_Pseudo4/exp
run 29_v2_01-pseudo.ipynb
run 29_v2_02-pseudo.ipynb
run 29_vl_01-pseudo.ipynb
```


13. train with 2nd round pseudo-label, then finetune on gt label of this competition
```
cd 34_RNN2
run code/ensemble_to_create_pseudo_label.ipynb
run exp/34_v2_02.ipynb
run exp/34_vl_01.ipynb
```


14. single model stacking for models trained with 1st round pseudo-label
```
cd 999_Stacking
run 999_v1_22/catboost.ipynb
run 999_v1_23/catboost.ipynb
```


15. convert oof data format for 4 models as follows.
```
cd 34_RNN2/exp/result
run 34_v2_02/converter.ipynb
run 34_vl_01/converter.ipynb
```

```
cd 999_Stacking
run 999_v1_22/result/converter.ipynb
run 999_v1_23/result/converter.ipynb
```

16. find an optimal weights for weighted averaging of 4 models as follows.
```
cd 99_Ensemble/99_v1_07/
run 99_v1_07_04.ipynb
```

