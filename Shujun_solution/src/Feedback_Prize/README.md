# Feedback-Prize---Predicting-Effective-Arguments

## Data preparation

I've compiled all data needed to run my solution into a single zip file to download from Kaggle:


```
kaggle datasets download -d shujun717/feedback-prize-effectiveness-shujun-input
```

then:

```
mkdir input
unzip feedback-prize-effectiveness-shujun-input.zip -d input/
```

Now you have all the data needed to run my solution.


# Reproducing multi-round pl model training

In total, I ran 5 rounds of pl model training for my ensemble in our final solution

## 1st round training

1. ```python write_scripts.py``` to create necessary sh scripts to run 6 fold training concurrently. If you don't have 6 gpus, you would need to remove the nohup commands and run training one fold at a time. You GPU will need to have 48 gb of vram for training to fit on a single gpu. Otherwise, add --gradient_checkpointing_enable to the command line arguments

2. ```bash run_folds_pl.sh``` to run training for first round models

3. ```bash get_pl_set.sh``` to get pl labels following first round training

4. Now run stacking code with the following commands:
```
bash scripts/stacking_1st_rd.sh
```

5. ensemble pl labels:
```
python scripts/ensemble.py
```

## 2nd round training

1. ```python write_scripts_2nd_rd.py``` to create necessary sh scripts to run 6 fold training concurrently. If you don't have 6 gpus, you would need to remove the nohup commands and run training one fold at a time. You GPU will need to have 48 gb of vram for training to fit on a single gpu. Otherwise, add --gradient_checkpointing_enable to the command line arguments

2. ```bash run_folds_pl.sh``` to run training for first round models

3. ```bash get_pl_set.sh``` to get pl labels following first round training

4. Now run stacking code with the following commands:
```
bash scripts/stacking_2nd_rd.sh
```

5. ensemble pl labels:
```
python scripts/ensemble_2nd_rd.py
```

## 3rd round training

1. ```python write_scripts_3rd_rd.py``` to create necessary sh scripts to run 6 fold training concurrently. If you don't have 6 gpus, you would need to remove the nohup commands and run training one fold at a time. You GPU will need to have 48 gb of vram for training to fit on a single gpu. Otherwise, add --gradient_checkpointing_enable to the command line arguments

2. ```bash run_folds_pl.sh``` to run training for first round models

3. ```bash get_pl_set.sh``` to get pl labels following first round training

4. Now run stacking code with the following commands:
```
bash scripts/stacking_4th_rd.sh
```

5. ensemble pl labels:
```
python scripts/ensemble_4th_rd.py


## 4th round training

1. ```python write_scripts_3rd_rd.py``` to create necessary sh scripts to run 6 fold training concurrently. If you don't have 6 gpus, you would need to remove the nohup commands and run training one fold at a time. You GPU will need to have 48 gb of vram for training to fit on a single gpu. Otherwise, add --gradient_checkpointing_enable to the command line arguments

2. ```bash run_folds_pl.sh``` to run training for first round models

3. ```bash get_pl_set.sh``` to get pl labels following first round training

4. Now run stacking code with the following commands:
```
bash scripts/stacking_4th_rd.sh
```

5. ensemble pl labels:
```
python scripts/ensemble_4th_rd.py
