import os
import time
import math
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pathlib import Path
from gflownet.models.config import TrainConfig
from gflownet.data.config import DataConfig

from gflownet.models.Transformer.model import Transformer
from gflownet.models.Transformer.preprocess import make_counter, make_transforms, make_dataloader
from gflownet.models.Transformer.utils import TransformerLR, tally_parameters, EarlyStopping, AverageMeter, accuracy, torch_fix_seed

import datetime
import os
import argparse

date = datetime.datetime.now().strftime('%Y%m%d')

# torch_fix_seed()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backward', '-b', action='store_true', help='backward policy')
    args = parser.parse_args()
    backward = args.backward
    if backward:
        print('training backward policy')
    else:
        print('training forward policy')
    cfg = TrainConfig()
    data_cfg = DataConfig()
    
    wandb.init(project='cTransformer_py3.11',
               config={
                   'backward': backward,
                   'd_Transformer': cfg.d_Transformer,
                   'num_encoder_layers': cfg.num_encoder_layers,
                   'num_decoder_layers': cfg.num_decoder_layers,
                   'nhead': cfg.nhead,
                   'dropout_Transformer': cfg.dropout_Transformer,
                   'dim_ff': cfg.dim_ff,
                   'batch_size_Transformer': cfg.batch_size_Transformer,
               }
               )
    model_cfg = cfg
    print(model_cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = Path(__file__).parent / 'ckpts' / f'checkpoints_{date}'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'{ckpt_dir} was created.')
    
    if not backward:
        data_dict = make_counter(
            src_train_path=data_cfg.USPTO_path / 'src_train.txt',
            tgt_train_path=data_cfg.USPTO_path / 'tgt_train.txt',
            src_valid_path=data_cfg.USPTO_path / 'src_valid.txt',
            tgt_valid_path=data_cfg.USPTO_path / 'tgt_valid.txt'
        )
    else:
        data_dict = make_counter(
            src_train_path=data_cfg.USPTO_path / 'src_train_product.txt',
            tgt_train_path=data_cfg.USPTO_path / 'tgt_train_reactant.txt',
            src_valid_path=data_cfg.USPTO_path / 'src_valid_product.txt',
            tgt_valid_path=data_cfg.USPTO_path / 'tgt_valid_reactant.txt'
        )
    
    print('making dataloader...')
    src_transforms, tgt_transforms, v = make_transforms(data_dict=data_dict, make_vocab=True, vocab_load_path=None)
    train_dataloader, valid_dataloader = make_dataloader(datasets=data_dict['datasets'], src_transforms=src_transforms,
                                                         tgt_transforms=tgt_transforms,batch_size=model_cfg.batch_size_Transformer)
    print('max length of src sentence:', data_dict['src_max_len'])
    d_model = model_cfg.d_Transformer
    nhead = model_cfg.nhead
    dropout = model_cfg.dropout_Transformer
    dim_ff = model_cfg.dim_ff
    num_encoder_layers = model_cfg.num_encoder_layers
    num_decoder_layers = model_cfg.num_decoder_layers
    model = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_ff,vocab=v, dropout=dropout, device=device).to(device)
    cudnn.benchmark = True
    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.benchmark = True
    
    # count the number of parameters.
    n_params, enc, dec = tally_parameters(model)
    print('encoder: %d' % enc)
    print('decoder: %d' % dec)
    print('* number of parameters: %d' % n_params)
    
    lr = model_cfg.lr_Transformer
    betas = model_cfg.betas_Transformer
    patience = 20
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = TransformerLR(optimizer, warmup_epochs=8000)
    label_smoothing = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                    reduction='none',
                                    ignore_index=v['<pad>']
                                    )
    earlystopping = EarlyStopping(patience=patience, ckpt_dir=ckpt_dir)
    
    step_num = 500000
    log_interval_step = 1000
    valid_interval_steps = 10000
    save_interval_steps = 10000
    accum_count = 1
    
    valid_len = 0
    for _, d in enumerate(valid_dataloader):
        valid_len += len(d[0])
    
    step = 0
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(data_dict['tgt_max_len']-1).to(device)
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    accum_loss = 0
    model.train()
    start_time = time.time()
    print('start training...')
    while step < step_num:
        for i, data in enumerate(train_dataloader):
            src, tgt = data[0].to(device).permute(1, 0), data[1].to(device).permute(1, 0)
            tgt_input = tgt[:-1, :] # (seq, batch)
            tgt_output = tgt[1:, :] # shifted right
            with torch.amp.autocast('cuda'):
                outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
                                src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True) # out: (seq_length, batch_size, vocab_size)
                loss = (criterion(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1)).sum() / len(data[0])) / accum_count
            scaler.scale(loss).backward()
            accum_loss += loss.detach().item()
            if ((i + 1) % accum_count == 0) or ((i + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                total_loss += accum_loss
                accum_loss = 0
            
            if (step + 1) % log_interval_step == 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval_step
                ppl = math.exp(cur_loss)
                end_time = time.time()
                wandb.log({
                    'step': step+1,
                    'lr': lr,
                    'loss': cur_loss,
                    'ppl': ppl,
                    'time_per_step': end_time - start_time
                })
                print(f'| step {step+1} | lr {lr:03.5f} | loss {cur_loss:5.5f} | ppl {ppl:8.5f} | time per {log_interval_step} step {end_time - start_time:3.1f}|')
                total_loss = 0
                start_time = time.time()
            
            # validation step
            if (step + 1) % valid_interval_steps == 0:
                model.eval()
                top1 = AverageMeter()
                perfect_acc_top1 = AverageMeter()
                eval_total_loss = 0.
                with torch.no_grad():
                    for val_i, val_data in enumerate(valid_dataloader):
                        src, tgt = val_data[0].to(device).permute(1, 0), val_data[1].to(device).permute(1, 0)
                        tgt_input = tgt[:-1, :]
                        tgt_output = tgt[1:, :]
                        outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
                                        src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True)
                        tmp_eval_loss = criterion(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1)).sum() / len(val_data[0])
                        eval_total_loss += tmp_eval_loss.detach().item()
                        partial_top1, perfect_acc = accuracy(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1), batch_size=tgt_output.size(1), v=v)
                        top1.update(partial_top1, src.size(1))
                        perfect_acc_top1.update(perfect_acc, src.size(1))
                    eval_loss = eval_total_loss / (val_i + 1)
                    print(f'validation step {step+1} | validation loss {eval_loss:5.5f} | partial top1 accuracy {top1.avg:.3f} | perfect top1 accuracy {perfect_acc_top1.avg:.3f}')
                    if (step + 1) % save_interval_steps == 0:
                        earlystopping(val_loss=eval_loss, step=step, optimizer=optimizer, cur_loss=cur_loss, model=model)
                    wandb.log({
                        'step': step,
                        'validation_loss': eval_loss,
                        'partial_top1_accuracy': top1.avg,
                        'perfect_top1_accuracy': perfect_acc_top1.avg
                    })
                model.train()
                start_time = time.time()
            if earlystopping.early_stop:
                print('Early Stopping!')
                break
        if earlystopping.early_stop:
            break
            
def main():
    train()


if __name__ == '__main__':
    main()