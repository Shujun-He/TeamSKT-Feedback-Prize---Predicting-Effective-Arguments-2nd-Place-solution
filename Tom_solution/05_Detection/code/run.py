import time
import pandas as pd
import numpy as np
import gc
from os.path import join as opj
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import TextSpanDetector, DatasetTrain, CustomCollator, TYPE2LABEL
from transformers import AutoTokenizer
from awp import AWP

def run(args, trn_df, val_df, pseudo_df=None):
    output_path = opj(f'./result', args.version)
    if True:
        if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
            from transformers.models.deberta_v2 import DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)

        trn_dataset = DatasetTrain(
            trn_df,
            text_dir=args.train_text_dir,
            tokenizer=tokenizer,
            mask_prob=args.mask_prob,
            mask_ratio=args.mask_ratio,
        )
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=args.trn_batch_size,
            shuffle=True,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
        )
        
        val_dataset = DatasetTrain(
            val_df,
            text_dir=args.train_text_dir,
            tokenizer=tokenizer,
            mask_prob=0,
            mask_ratio=0,
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
        num_trn_dataset = len(trn_dataset)
        num_train_steps = int(num_trn_dataset / args.trn_batch_size / args.accumulate_grad_batches * args.epochs)
        print('num_train_steps = ', num_train_steps)
        model_pretraining = None
        model = TextSpanDetector(
            args.model,
            tokenizer,
            num_classes=args.num_classes,
            dynamic_positive=True,
            with_cp=(args.check_pointing=='true'),
            hidden_dropout_prob=args.hidden_drop_prob, 
            p_drop=args.p_drop,
            learning_rate=args.lr,
            head_learning_rate=args.head_lr,
            num_train_steps=num_train_steps,
            warmup_ratio=args.warmup_ratio,
            model_pretraining=model_pretraining,
            rnn=args.rnn,
            loss=args.loss,
            head=args.head,
            msd=args.msd,
            multi_layers=args.multi_layers,
            aug=args.aug,
            mixup_alpha=args.mixup_alpha,
            aug_stop_epoch=args.aug_stop_epoch,
            p_aug=args.p_aug,
            adv_sift=args.adv_sift,
            l2norm=args.l2norm,
            weight_decay=args.weight_decay,
            freeze_layers=args.freeze_layers,
            mt=args.mt,
            w_mt=args.w_mt,
            scheduler=args.scheduler,
            num_cycles=args.num_cycles,
        )
        
        model = model.cuda()
        if args.pretrain_path != 'none':
            model.load_state_dict(torch.load(args.pretrain_path))
            
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16=='true'))
        
        [optimizer], [scheduler] = model.configure_optimizers()
        scheduler = scheduler['scheduler']
        
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
                scheduler.step()
                continue
                
            print('\nlr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            trn_loss = 0
            trn_obj_loss = 0
            trn_reg_loss = 0
            trn_cls_loss = 0
            trn_score = 0
            counter = 0
            tk0 = tqdm(trn_dataloader, total=int(len(trn_dataloader)))
            optimizer.zero_grad()
            trn_preds = []
            trn_trues = []
            
            if args.awp=='true':
                awp = AWP(
                    model,
                    criterion=nn.CrossEntropyLoss(reduction='none'), 
                    optimizer=optimizer,
                    apex=(args.fp16=='true'),
                    adv_lr=args.awp_lr, 
                    adv_eps=args.awp_eps
                )
            
            for i,data in enumerate(tk0):
                batch = trn_dataloader.batch_size
                model.train()
                with torch.cuda.amp.autocast(enabled=(args.fp16=='true')):
                    trn_res = model.training_step(data, trn_df, test_score_thr=args.test_score_thr)
                    loss = trn_res['loss']
                    
                    if args.accumulate_grad_batches > 1:
                        loss = loss / args.accumulate_grad_batches
                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                    
                    trn_loss += loss.item() * batch * args.accumulate_grad_batches
                    trn_obj_loss += trn_res['obj_loss']  * batch
                    trn_reg_loss += trn_res['reg_loss']  * batch
                    trn_cls_loss += trn_res['cls_loss']  * batch
                    trn_score += trn_res['score'] * batch
                    
                    if args.awp=='true':
                        if epoch >= args.awp_start_epoch:
                            loss = awp.attack_backward(data, trn_df, test_score_thr=args.test_score_thr)
                            if args.accumulate_grad_batches > 1:
                                loss = loss / args.accumulate_grad_batches
                            scaler.scale(loss).backward()
                            awp._restore()
                            trn_loss += loss.item() * batch * args.accumulate_grad_batches

                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                
                counter += 1
                tk0.set_postfix(loss=(trn_loss / (counter * batch) ), score=(trn_score / (counter * batch) ))
                
                # eval
                if args.eval_step!=-1 and (i+1)%args.eval_step==0:
                    model.eval()
                    outputs = []
                    for i,data in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
                        #release GPU memory cache
                        del data
                        torch.cuda.empty_cache()
                        gc.collect()
                    val_res = model.validation_epoch_end(outputs, add_epoch=False)
                    val_loss, val_score = val_res['loss'], val_res['score']

                    #monitering
                    print('\nepoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                        epoch, 
                        trn_loss / (counter * trn_dataloader.batch_size), 
                        val_loss, 
                        trn_score / (counter * trn_dataloader.batch_size),
                        val_score
                    ))
                    
                    print('trn_obj_loss={:.4f}, trn_reg_loss={:.4f}, trn_cls_loss={:.4f}'.format(
                        trn_obj_loss / (counter * trn_dataloader.batch_size),
                        trn_reg_loss / (counter * trn_dataloader.batch_size), 
                        trn_cls_loss / (counter * trn_dataloader.batch_size)
                    ))
                    
                    print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}'.format(
                        val_res['obj_loss'], val_res['reg_loss'], val_res['cls_loss']
                    ))
                          
                    if args.slack_url!='none':
                        from utils import post_message
                        post_message(name='bot',
                                     message='epoch {:.0f}: trn_loss = {:.4f}, val_loss={:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                                         epoch, trn_loss, val_loss, trn_score, val_score), 
                                     incoming_webhook_url=args.slack_url)
                    if args.early_stopping=='true':
                        if val_loss < val_loss_best: #val_score > val_score_best:
                            val_score_best = val_score #update
                            val_loss_best  = val_loss #update
                            epoch_best     = epoch #update
                            counter_ES     = 0 #reset
                            torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                            print('model (best loss) saved')
                        else:
                            counter_ES += 1
                        if counter_ES > args.patience:
                            print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                                epoch_best, val_loss_best, val_score_best))
                            break
                    else:
                        torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save

                    if val_score > val_score_best2:
                        val_score_best2 = val_score #update
                        torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}.pth')) #save
                        print('model (best score) saved')
                
            trn_loss = trn_loss / len(trn_dataset)
            trn_score = trn_score / len(trn_dataset)
            
            #release GPU memory cache
            del data, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # save
            torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_epoch{epoch}.pth')) #save

            #eval
            model.eval()
            outputs = []
            tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
            for i,data in enumerate(tk1):
                with torch.no_grad():
                    outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
                #release GPU memory cache
                del data
                torch.cuda.empty_cache()
                gc.collect()
            val_res = model.validation_epoch_end(outputs)
            val_loss, val_score = val_res['loss'], val_res['score']
            
            #monitering
            print('epoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                epoch, 
                trn_loss,
                val_loss,
                trn_score, 
                val_score,
            ))
            
            print('trn_obj_loss={:.4f}, trn_reg_loss={:.4f}, trn_cls_loss={:.4f}'.format(
                trn_obj_loss / len(trn_dataset),
                trn_reg_loss / len(trn_dataset), 
                trn_cls_loss / len(trn_dataset)
            ))
            
            print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}'.format(
                val_res['obj_loss'], val_res['reg_loss'], val_res['cls_loss']
            ))
                  
            if args.slack_url!='none':
                from utils import post_message
                post_message(name='bot',
                             message='epoch {:.0f}: trn_loss = {:.4f}, val_loss={:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                                 epoch, trn_loss, val_loss, trn_score, val_score), 
                             incoming_webhook_url=args.slack_url)
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if args.early_stopping=='true':
                if val_loss < val_loss_best: #val_score > val_score_best:
                    val_score_best = val_score #update
                    val_loss_best  = val_loss #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
                    torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > args.patience:
                    print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                        epoch_best, val_loss_best, val_score_best))
                    break
            else:
                torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
               
            if val_score > val_score_best2:
                val_score_best2 = val_score #update
                torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}.pth')) #save
                print('model (best score) saved')
                
        #best model
        if args.early_stopping=='true' and counter_ES<=args.patience:
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, val_loss_best, val_score_best))
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')