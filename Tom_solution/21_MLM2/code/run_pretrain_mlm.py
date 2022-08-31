import time
import pandas as pd
import numpy as np
import gc
from os.path import join as opj
import pickle
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models_pretrain_generator import DatasetTrain, CustomCollator, Model as Generator
discourse_type_list = [
    'Lead',
    'Position',
    'Claim',
    'Counterclaim',
    'Rebuttal',
    'Evidence',
    'Concluding Statement'
]

def run(args, trn_df, val_df, pseudo_df=None):
    output_path = opj(f'./result', args.version)
    if True:
        if 'deberta-v2' in args.generator or 'deberta-v3' in args.generator:
            from transformers.models.deberta_v2 import DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast.from_pretrained(args.generator, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in discourse_type_list]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.generator, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in discourse_type_list]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)

        # dataset
        trn_dataset = DatasetTrain(
            trn_df,
            tokenizer, 
            ratio_masking=args.ratio_masking,
            max_length=args.max_length,
            text_dir=args.text_dir,
            mode='train'
        )
        val_dataset = DatasetTrain(
            val_df, 
            tokenizer, 
            ratio_masking=args.ratio_masking, 
            max_length=args.max_length,
            text_dir=args.text_dir,
            mode='valid'
        )
    
        # dataloader
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=args.trn_batch_size,
            shuffle=True,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
        
        #model
        num_train_steps = int(len(trn_dataset) / args.trn_batch_size / args.accumulate_grad_batches * args.epochs)
        model_pretraining = None
        model_gen = Generator(args.generator, 
                              tokenizer,
                              num_labels=len(tokenizer), 
                              hidden_dropout_prob=args.hidden_drop_prob, 
                              p_drop=args.p_drop,
                              learning_rate=args.lr,
                              head_learning_rate=args.head_lr,
                              num_train_steps=num_train_steps,
                              warmup_ratio=args.warmup_ratio,
                              ratio_masking=args.ratio_masking,
                              freeze_layers=args.freeze_layers,
                              scheduler=args.scheduler,
                              num_cycles=args.num_cycles,
                              with_cp=(args.check_pointing=='true'),
                              window_size=args.window_size,
                              inner_len=args.inner_len,
                              edge_len=args.edge_len,
                             )
        model_gen = model_gen.cuda()
        if args.pretrain_path != 'none':
            model_gen.load_state_dict(torch.load(args.pretrain_path))
        
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16=='true'))
        
        [optimizer_gen], [scheduler_gen] = model_gen.configure_optimizers()
        scheduler_gen = scheduler_gen['scheduler']
        
        #training
        val_score_best  = -1e+99
        val_score_best2 = -1e+99
        val_loss_best   = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        start_time = time.time()
        for epoch in range(1, args.epochs+1):
            if epoch > args.stop_epoch:
                break
            if epoch < args.restart_epoch:
                print('epoch = ', epoch)
                tk0 = tqdm(trn_dataloader, total=int(len(trn_dataloader)))
                for i,data in enumerate(tk0):
                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scheduler_gen.step()
                continue
                
            print('lr : ', [ group['lr'] for group in optimizer_gen.param_groups ])
            
            #train
            trn_loss_gen = 0
            trn_score_gen = 0
            trn_loss = 0
            trn_score = 0
            counter = 0
            tk0 = tqdm(trn_dataloader, total=int(len(trn_dataloader)))
            optimizer_gen.zero_grad()
            for i,data in enumerate(tk0):
                model_gen.train()
                batch = len(data['data_id'])
                
                with torch.cuda.amp.autocast(enabled=(args.fp16=='true')):
                    # MLM task for generator
                    loss_gen, score_gen, output_data_gen = model_gen.training_step(data)
                        
                    if args.accumulate_grad_batches > 1:
                        loss_gen = loss_gen / args.accumulate_grad_batches
                        
                    #scaler.scale(args.rtd_lambda * loss + loss_gen).backward()
                    scaler.scale(loss_gen).backward()
                        
                    grad_norm_gen = torch.nn.utils.clip_grad_norm_(model_gen.parameters(), args.gradient_clip_val)
                
                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scaler.step(optimizer_gen)
                        scaler.update()
                        optimizer_gen.zero_grad()
                        scheduler_gen.step()
                
                trn_loss_gen += loss_gen.item() * batch * args.accumulate_grad_batches
                trn_score_gen += score_gen * batch
                counter  += 1
                tk0.set_postfix(
                    loss=(trn_loss_gen / (counter * trn_dataloader.batch_size) ),
                    score=(trn_score_gen / (counter * trn_dataloader.batch_size))
                )
                
                # eval
                if args.eval_step!=-1 and (i+1)%args.eval_step==0:
                    model_gen.eval()
                    outputs_gen = []
                    for i,data in enumerate(val_dataloader):
                        with torch.no_grad():
                            
                            # MLM task for generator
                            output_gen, output_data_gen = model_gen.validation_step(data)
                            outputs_gen.append(output_gen)
                            
                        #release GPU memory cache
                        del data
                        torch.cuda.empty_cache()
                        gc.collect()
                    val_loss_gen, val_score_gen = model_gen.validation_epoch_end(outputs_gen)

                    #monitering
                    print('\nepoch {:.0f}: trn_loss_gen = {:.4f}, val_loss_gen = {:.4f}, trn_score_gen = {:.4f}, val_score_gen = {:.4f}'.format(
                        epoch, 
                        trn_loss_gen / (counter * trn_dataloader.batch_size), 
                        val_loss_gen,
                        trn_score_gen / (counter * trn_dataloader.batch_size),
                        val_score_gen)
                         )
                    if args.early_stopping=='true':
                        if val_loss_gen < val_loss_best: #val_score > val_score_best:
                            val_score_best = val_score_gen #update
                            val_loss_best  = val_loss_gen #update
                            epoch_best     = epoch #update
                            counter_ES     = 0 #reset
                            torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                            print('model_gen (best loss) saved')
                        else:
                            counter_ES += 1
                        if counter_ES > args.patience:
                            print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                                epoch_best, val_loss_best, val_score_best))
                            break
                    else:
                        torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                    if val_score_gen > val_score_best2:
                        val_score_best2 = val_score_gen #update
                        torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}.pth')) #save
                        print('model_gen (best score) saved')
                
            trn_loss_gen = trn_loss_gen / len(trn_dataset)
            trn_score_gen = trn_score_gen / len(trn_dataset)
            
            #release GPU memory cache
            del data#, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # save model
            torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}_epoch{epoch}.pth')) #save

            #eval
            model_gen.eval()
            outputs_gen = []
            tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
            for i,data in enumerate(tk1):
                with torch.no_grad():
                    
                    # MLM task for generator
                    output_gen, output_data_gen = model_gen.validation_step(data)
                    outputs_gen.append(output_gen)
                            
                #release GPU memory cache
                del data
                torch.cuda.empty_cache()
                gc.collect()
            val_loss_gen, val_score_gen = model_gen.validation_epoch_end(outputs_gen)
            
            #monitering
            print('\nepoch {:.0f}: trn_loss_gen = {:.4f}, val_loss_gen = {:.4f}, trn_score_gen = {:.4f}, val_score_gen = {:.4f}'.format(
                epoch, 
                trn_loss_gen,
                val_loss_gen,
                trn_score_gen,
                val_score_gen)
                 )
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if args.early_stopping=='true':
                if val_loss_gen < val_loss_best: #val_score > val_score_best:
                    val_score_best = val_score_gen #update
                    val_loss_best  = val_loss_gen #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
                    torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                    print('model_gen (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > args.patience:
                    print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                        epoch_best, val_loss_best, val_score_best))
                    break
            else:
                torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
            
            if val_score_gen > val_score_best2:
                val_score_best2 = val_score_gen #update
                torch.save(model_gen.state_dict(), opj(output_path,f'model_gen_seed{args.seed}_fold{args.fold}.pth')) #save
                print('model_gen (best score) saved')
                
        #best model
        if args.early_stopping=='true' and counter_ES<=args.patience:
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, val_loss_best, val_score_best))
        
        #del model, model_gen
        del model_gen
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')