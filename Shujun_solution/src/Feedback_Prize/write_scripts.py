import os

os.system('mkdir scripts')

N_FOLDS=6

with open('run_folds_pl.sh','w+') as f:
    for i in range(N_FOLDS):
        with open(f'scripts/{i}_pl.sh','w+') as f2:
            #for j in range(10):
            f2.write(f'python run_pl.py --fold {i} --gpu_id {i} \
--downloaded_model_path ../../input/tascj0_deberta_v3_large --model_name deberta-v3-large  \
--gradient_accumulation_steps 8 --batch_size 4 --nfolds {N_FOLDS} \
--experiment_name deberta_v3_large_tascj0 --rnn GRU --lr_scale 0.75 --epochs 3 --pl_epochs 0 --max_grad_norm 1 \
--lr_schedule "linear anneal" --weight_decay 1e-2 --max_len 1280 --do_val \
--train_csv_path ../../input/feedback-prize-effectiveness/train.csv \
--full_text_path ../../input/feedback-prize-effectiveness/train \
--train_pl_csv_path ../../input/train_pl.csv \
--do_val --window_size 1280 \n')

            f2.write(f'python run_pl.py --fold {i} --gpu_id {i} \
--downloaded_model_path ../../input/tascj0_deberta_v2_xlarge --model_name deberta-v2-xlarge  \
--gradient_accumulation_steps 4 --batch_size 2 --nfolds {N_FOLDS} \
--experiment_name deberta_v2_xlarge_tascj0 --rnn GRU --lr_scale 0.2 --epochs 3 --pl_epochs 0 --max_grad_norm 1 \
--lr_schedule "linear anneal" --weight_decay 1e-2 --max_len 1280 --do_val \
--train_csv_path ../../input/feedback-prize-effectiveness/train.csv \
--full_text_path ../../input/feedback-prize-effectiveness/train \
--train_pl_csv_path ../../input/train_pl.csv \
--do_val --window_size 1280 \n')


        f.write(f'nohup bash scripts/{i}_pl.sh > {i}.out & \n')



with open('get_pl_set.sh','w+') as f:
    for i in range(N_FOLDS):

        with open(f'scripts/{i}_get_pl.sh','w+') as f3:

            f3.write(f'python get_pl_labels.py --fold {i} --gpu_id {i+1} --hidden_state_dimension 1024 \
--downloaded_model_path ../../input/deberta-v3-large --model_name deberta-v3-large \
--gradient_accumulation_steps 8 --batch_size 16 --nfolds {N_FOLDS} \
--experiment_name deberta_v3_large_tascj0 --cpc_level 4 --lr_scale 0.75 --epochs 3 \
--lr_schedule "linear anneal" --loss CrossEntropyLoss --split_by anchor --weight_decay 1e-2 --max_len 1280 --do_val \
--cpc_path ../../input/cpc-scheme-dataframe.json \
--train_csv_path ../../input/train_pl.csv \
--full_text_path ../../../Feedback_Prize/input/train \
--do_val --window_size 1280 --num_workers 16 \n')

            f3.write(f'python get_pl_labels.py --fold {i} --gpu_id {i+1} --hidden_state_dimension 1024 \
--downloaded_model_path ../../input/deberta-v2-xlarge --model_name deberta-v2-xlarge \
--gradient_accumulation_steps 8 --batch_size 16 --nfolds {N_FOLDS} \
--experiment_name deberta_v2_xlarge_tascj0 --cpc_level 4 --lr_scale 0.75 --epochs 3 \
--lr_schedule "linear anneal" --loss CrossEntropyLoss --split_by anchor --weight_decay 1e-2 --max_len 1280 --do_val \
--cpc_path ../../input/cpc-scheme-dataframe.json \
--train_csv_path ../../input/train_pl.csv \
--full_text_path ../../../Feedback_Prize/input/train \
--do_val --window_size 1280 --num_workers 16 \n')

        f.write(f'nohup bash scripts/{i}_get_pl.sh > {i}.out & \n')
