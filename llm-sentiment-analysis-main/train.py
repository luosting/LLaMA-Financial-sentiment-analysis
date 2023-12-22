"""
Llama2 pretrain codes.
"""

import json
import os
import sys
import time
import datetime
import numpy as np
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import multiprocessing as mp
import torch
import pickle
import io
import PIL
import matplotlib.pyplot as plt
import queue
import torchvision
import warnings
from fairscale.nn.model_parallel.initialize import (
    get_data_parallel_rank,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_data_parallel_group,
    get_model_parallel_group,
    get_data_parallel_world_size,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama_train.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
import argparse
from dataset import PretrainData, PrefetchDataLoader, SFTData
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.cuda.amp import GradScaler
from fairscale.optim.oss import OSS
import random
import string
import tqdm
from functools import partial
from cosine_annealing_warmup_scheduler import CosineAnnealingWarmupRestarts
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from utils import measure_peak_memory, dist_print
import socket

warnings.filterwarnings("ignore", category=UserWarning)

def log(msg, logfile, rank):
    if rank == 0:
        print(msg)
        with open(logfile, 'a+') as fp:
            fp.write(msg+'\n')

def save_checkpoint(checkpoint_dir, model, optimizer, rope_freqs=None, **kwargs):
    if os.path.isdir(checkpoint_dir) == False:
        os.system(f'mkdir -p {checkpoint_dir}')
    d = {}
    d['rope.freqs'] = rope_freqs
    for k, v in model.state_dict().items():
        if k.startswith('module.'): k = k[7:]
        d[k] = v
    # save model weights.
    if get_data_parallel_rank() == 0:
        filename = f'{checkpoint_dir}/model-rank-{get_model_parallel_rank()}.pth'
        torch.save(d, filename)

def init_model_weights(model):
    for name, param in model.named_parameters():
        if name.find('norm.weight') > -1:
            param.data[:] = 0.5 # based on llama7b pretrained.
        elif name.find('.weight') > -1:
            torch.nn.init.xavier_normal_(param.data)
        elif name.find('.bias') > -1:
            param.data *= 0
        else:
            raise ValueError(name)

def train(args):

    # init dist.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        initialize_model_parallel(1)

    # env
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    assert local_rank > -1
    assert WORLD_SIZE > 0
    global_rank = torch.distributed.get_rank()

    #
    if global_rank > 0:
        # disable logging on worker machines.
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        pass

    # sanity checks
    if args.global_batch_size == -1:
        # not using gradient accumulation.
        args.global_batch_size = args.micro_batch_size * get_data_parallel_world_size()
    else:
        # may use gradient accumulation.
        assert args.global_batch_size % (get_data_parallel_world_size() * args.micro_batch_size) == 0, f'{args.global_batch_size=}, {args.micro_batch_size=}, {get_data_parallel_world_size()}'

    # set cuda device for this process.
    torch.cuda.set_device(local_rank)

    # half tensor
    if args.tensor_type == "BFLOAT16":
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor) # FloatTensor, HalfTensor, BFloat16Tensor
    elif args.tensor_type == "FLOAT16":
        torch.set_default_tensor_type(torch.cuda.HalfTensor) # FloatTensor, HalfTensor, BFloat16Tensor
    elif args.tensor_type == "FLOAT32":
        torch.set_default_tensor_type(torch.cuda.FloatTensor) # FloatTensor, HalfTensor, BFloat16Tensor
    else:
        raise ValueError(args.tensor_type)

    # logging
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    args.save_ckpt_dir = f'{args.save_ckpt_dir}/{time_string}'
    if global_rank == 0:
        if not os.path.isdir(f'{args.save_ckpt_dir}'):
            os.system(f'mkdir -p {args.save_ckpt_dir}')
        assert os.path.isdir(f'{args.save_ckpt_dir}'), args.save_ckpt_dir
    logfile = f'{args.save_ckpt_dir}/log.txt'
    loginfo = partial(log, rank=global_rank, logfile=logfile)
    loginfo(f'Logging into {logfile}')

    # model arguments
    with open(f"{args.load_ckpt_dir}/params.json", "r") as f:
        params = json.loads(f.read())
        loginfo(f'params: {params}')
    model_args = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.global_batch_size,
        activation=args.activation,
        n_offload_skips=args.n_offload_skips,
        **params)

    # tokenizer
    tokenizer_path='checkpoints/llama2_7b_base/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    # model
    model = Transformer(model_args)
    loginfo(f"Created transormer: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB")

    # init model weights
    if False:
        # random init
        init_model_weights(model)
        ckpt_data = {}
    else:
        # loading checkpoints
        t0 = time.time()
        print('Loading checkpoinit')
        for ckpt_dir, m in [(args.load_ckpt_dir, model), (None, None)]:
            if not m: continue
            checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
            assert args.model_parallel_size == len(
                checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {args.model_parallel_size}"
            ckpt_path = checkpoints[get_model_parallel_rank()]
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            """
            assert False, ckpt_data.keys()
            if 'model' in ckpt_data:
                checkpoint = ckpt_data['model']
                if not args.resume: ckpt_data = checkpoint # if not resume (e.g. from embedding), then train from begin.
            else:
                checkpoint = ckpt_data
            # fix key
            fix_checkpoint = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'): k = k[7:]
                fix_checkpoint[k] = v
                #if k.find('norm.weight') > -1:
                #    print(f'{k=} {v.mean()=} {v.sum()=} {v.numel()=} {v[0]=}')
            checkpoint = fix_checkpoint
            """
            if "rope.freqs" in ckpt_data:
                rope_freqs = ckpt_data['rope.freqs']
                del ckpt_data["rope.freqs"]
            else:
                rope_freqs = None
            m.load_state_dict(ckpt_data, strict=True)
        dt = time.time() - t0
        loginfo(f'Loaded checkpoint from {args.load_ckpt_dir=} with {dt=}s')

    # seed must be the same in all processes
    seed = 1234
    if 'seed' in ckpt_data: seed = ckpt_data['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # optimizer
    if 'base_optimizer_arguments' in ckpt_data:
        base_optimizer_arguments = ckpt_data['base_optimizer_arguments']
    else:
        base_optimizer_arguments = {"betas": (0.9, 0.999), "lr": args.max_lr,
                                    "weight_decay": 1e-3, "eps": 1e-8}
    base_optimizer = torch.optim.AdamW


    # full parameters training
    if args.train_ddp:
        optimizer = base_optimizer(
            model.parameters(),
            **base_optimizer_arguments)
    else:
        optimizer = OSS(
            model.parameters(),
            optim=base_optimizer,
            group=get_data_parallel_group(),
            **base_optimizer_arguments)
    loginfo(f"Created optimizer: {base_optimizer_arguments=}")


    # dataset
    if args.train_sft:
        # training with sft data
        data_file = 'data/train_data.txt'
        dataset = SFTData(
            seed,
            data_file,
            args.max_seq_len,
            tokenizer,
            num_epochs=5,
        )
    else:
        # training with pretrain data
        dataset = PretrainData(
            seed,
            args.tokens_dir,
            args.max_seq_len,
            tokenizer.pad_id,
        )

    # data loader
    data_loader = PrefetchDataLoader(dataset, args.max_seq_len, args.prefetch_factor, args.micro_batch_size)
    data_loader.start()
    if 'iter_steps' in ckpt_data:
        iter_steps = ckpt_data['iter_steps']
    else:
        iter_steps = None
    loginfo(f"Data loader: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB. resume_steps={iter_steps}")

    # scheduler
    warmup_steps = int((args.warmup_tokens_b*1e9)//args.global_batch_size//args.max_seq_len)
    first_cycle_steps = int((args.first_cycle_tokens_b*1e9)//args.global_batch_size//args.max_seq_len)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        gamma=0.5)

    # loss function
    criterion = torch.nn.CrossEntropyLoss(
        reduction="mean",
        ignore_index=tokenizer.pad_id)

    # ddp
    if args.train_ddp:
        model2 = DDP(model, find_unused_parameters=True)
    else:
        model = ShardedDDP(
            model,
            optimizer,
            reduce_buffer_size=4000000, # 1M to 8M is usually reasonable, 0 if disabled (default). 
            process_group=get_data_parallel_group(),
            warn_on_trainable_params_changed=False,
        )
    loginfo(f"Wrap ShardedDataParallel: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB")

    # build gradient tracker map
    gradient_tags = ['tok_embeddings.weight', 'output.weight'] +\
        [f'layers.{k}.attention.wq.weight' for k in range(0,100,5)] +\
        [f'layers.{k}.attention.wk.weight' for k in range(0,100,5)] +\
        [f'layers.{k}.attention.wv.weight' for k in range(0,100,5)] +\
        [f'layers.{k}.attention.wo.weight' for k in range(0,100,5)] +\
        [f'layers.{k}.feed_forward.w3.weight' for k in range(0,100,5)]
    gradient_map = {}
    index = 0
    for name, param in model.named_parameters():
        select = False
        for tag in gradient_tags:
            if name.find(tag) > -1:
                select = True
                break
        if select:
            gradient_map[name] = index
            index += 1

    # tensorboard
    if global_rank == 0:
        if not os.path.isdir(f'{args.save_ckpt_dir}/tb'):
            os.system(f'mkdir -p {args.save_ckpt_dir}/tb')
        layout = {
            "Gradients": {
                "GraidentL2Norm": ["Multiline", [f"GraidentL2Norm/{name}" for name, index in gradient_map.items()]],
            },
        }
        writer = SummaryWriter(f'{args.save_ckpt_dir}/tb')
        writer.add_custom_scalars(layout)


    # freeze backbone if training embedding only.

    # init training states
    num_grad_accumulate_steps = args.global_batch_size // (get_data_parallel_world_size() * args.micro_batch_size)
    loginfo(f'{num_grad_accumulate_steps=}, {args.micro_batch_size=}, {args.global_batch_size=}')
    iter_steps = 0
    batch_steps = 0
    optimizer.zero_grad()
    dummy_x = None
    dummy_y = None
    t_start = time.time()
    global_num_tokens_consumed = 0
    global_num_tokens_chinese = 0
    global_num_tokens_english = 0
    global_num_tokens_code = 0
    num_log_steps = 10
    num_nans = 0
    checkpoint_tokens_consumed = 0 # reset after each checkpoint is saved.
    statistics = torch.tensor([0, 0, 0, 0], dtype=torch.float) # (#tokens, loss, dt_data, num_nan, checksum_1, checksum_2, n_chinese, n_english, n_code, gradients)

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    # resume training if necessary
    if 'iter_steps' in ckpt_data:
        for step in tqdm.tqdm(list(range(ckpt_data['iter_steps'])), desc=f'resuming {iter_steps=}'):
            scheduler.step()
        iter_steps = ckpt_data['iter_steps']
        global_num_tokens_consumed = ckpt_data['global_num_tokens_consumed']
        loginfo(f'Resumed from {global_num_tokens_consumed=:,}')

    ### Start training
    loginfo(f'model_args: {model_args}')
    torch.cuda.reset_peak_memory_stats()
    dt_helpers = queue.Queue()
    dt_total = queue.Queue()
    torch.distributed.barrier()
    loginfo(f">>>Start training: model_parallel_size={get_model_parallel_world_size()} data_parallel_size={get_data_parallel_world_size()} micro_batch_size={args.micro_batch_size} global_batch_size={args.global_batch_size}")

    ####
    while True:
        dt_acc = 0
        t_step_start = time.time()
        # fetch next data.
        t0 = time.time()
        try:
            x, y, seq_len_list = next(data_loader)
            """
            if global_rank == 0:
                with open('data.txt', 'w+') as fp:
                    fp.write(f'{x=}\n{y=}')
            assert False
            """
            batch_steps += 1
        except StopIteration:
            # no more data: save checkpoint.
            tokens_b = int(np.round(100*float(global_num_tokens_consumed) / 1e9))/100.0 # e.g. 0.01/1.00
            checkpoint_dir = f'{args.save_ckpt_dir}/{tokens_b}B'
            parameters = {
                'global_num_tokens_consumed': global_num_tokens_consumed,
                'iter_steps': iter_steps,
                'base_optimizer_arguments': base_optimizer_arguments,
                'seed': seed
            }
            if local_rank == 0:
                save_checkpoint(checkpoint_dir, model, optimizer, rope_freqs=rope_freqs, **parameters)
                with open(f'{checkpoint_dir}/params.json', 'w+') as fp:
                    json.dump(params, fp)
                with open(f'{args.save_ckpt_dir}/args.json', 'w+') as fp:
                    json.dump(vars(args), fp)
                os.system(f'cp {tokenizer_path} {checkpoint_dir}')
                loginfo(f'\n\n==========================\nSaved checkpoint to {checkpoint_dir} at {global_num_tokens_consumed=:,}.\n')
            break
        dt_data = time.time() - t0
        dt_acc += dt_data
        if dummy_x is None:
            t_start_global = time.time()
            dummy_x = torch.clone(x)
            dummy_y = torch.clone(y)
            dummy_y[:,-1] = tokenizer.pad_id
        def execute_forward_backward(num_nans):
            dt_acc = 0
            t0 = time.time()
            initial_memory = torch.cuda.memory_allocated()
            dt_acc += time.time() - t0
            for s in seq_len_list: assert min(s) > 0, s
            assert x.max() < model_args.vocab_size, f'{x.max()=} vs {model_args.vocab_size=}'
            assert x.min() >= 0
            logits = model.forward(x, seq_len_list=seq_len_list) # (batch, seq, vocab)
            t0 = time.time()
            after_memory = torch.cuda.memory_allocated()
            activation_memory = after_memory - initial_memory - logits.element_size()*torch.numel(logits)
            activation_memory /= 1e9
            dt_acc += time.time() - t0
            loss = criterion(
                input=logits.view(-1, logits.size(-1)),
                target=y.view(-1))
            if not (torch.isnan(loss) or torch.isinf(loss)):
                lossval = loss.item() # loss of each process.
            if torch.isnan(loss) or torch.isinf(loss):
                # do a dummy forward instead.
                loginfo(f'nan/inf loss encountered and skipped!')
                logits = model.forward(dummy_x, seq_len_list=[[len(x)] for x in dummy_x])
                loss = criterion(
                    input=logits.transpose(1, 2),
                    target=dummy_y)
                num_nans += 1
                loss *= 0
                lossval = loss.item() # loss of each process.
            loss /= num_grad_accumulate_steps
            loss.backward()
            return (lossval, num_nans, activation_memory, dt_acc)
        ###
        iter_steps += 1
        if iter_steps % num_grad_accumulate_steps == 0:
            L, num_nans, activation_memory, _dt_acc = execute_forward_backward(num_nans)
            dt_acc += _dt_acc
            # clip gradients
            if args.grad_clip > 0: optimizer.clip_grad_norm(args.grad_clip)
            # update model parameters
            optimizer.step()
            t0 = time.time()
            scheduler.step()
            dt_acc += time.time() - t0
            optimizer.zero_grad()
            # batch_steps += 1
        else:
            # doing grad accumulation locally
            with model.no_sync():
                L, num_nans = execute_forward_backward(num_nans)
        # next step
        t_stat_start = time.time()
        #statistics[0] += torch.numel(x) - sum([s[-1] for s in seq_len_list]) # remove padding
        statistics[0] += torch.numel(x) # remove padding
        statistics[1] += L
        statistics[2] += dt_data
        statistics[3] += num_nans
        if iter_steps % num_log_steps == 0:
            # dt
            if dt_helpers.qsize() > 0:
                # time consumed for accessary helpers.
                l1 = []
                l2 = []
                while dt_helpers.qsize() > 0:
                    l1.append(dt_helpers.get())
                    l2.append(dt_total.get())
                mean_dt_helper = np.mean(l1)
                mean_dt_total = np.mean(l2)
            else:
                mean_dt_helper = 0
                mean_dt_total = 1.0
            torch.distributed.all_reduce(statistics) # accumulated over num_log_steps*model_parallel_size*data_parallel_size
            if iter_steps % 100 == 0:
                torch.distributed.barrier()
            statistics /= get_model_parallel_world_size() # all process = #data x #model
            s = statistics.cpu().numpy() # accumulated over num_log_steps * data_parallel_size
            total_token_consumed, total_loss, total_data, total_num_nans = s[:4]
            total_token_consumed = int(total_token_consumed)
            global_num_tokens_consumed += total_token_consumed
            checkpoint_tokens_consumed += total_token_consumed
            tkps = total_token_consumed / (time.time() - t_start)
            loss = total_loss / get_data_parallel_world_size() / num_log_steps
            dt_data = total_data / get_data_parallel_world_size() / num_log_steps
            current_lr = [f"{optimizer.param_groups[k]['lr']:.2E}"
                          for k in range(len(optimizer.param_groups))]
            peak_memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 3
            msg = f'{time_string} {loss=:.2f} {current_lr=} speed={int(tkps):,} elapsed={time.time()-t_start_global:.2f}s iters={batch_steps} {global_num_tokens_consumed=:,} {dt_data=:.2E} {total_num_nans=} {activation_memory=:.2f}G {mean_dt_helper=:.4f}s helper_time_ratio={mean_dt_helper/mean_dt_total:.4f}'
            # tensorboard
            if global_rank == 0:
                loginfo(msg)
                writer.add_scalar('Loss', loss, global_num_tokens_consumed)
                writer.add_scalar('#NAN', total_num_nans, global_num_tokens_consumed)
                for k in range(len(optimizer.param_groups)):
                    current_lr = optimizer.param_groups[k]['lr']
                    writer.add_scalar(f'LR/{k}', current_lr, global_num_tokens_consumed)
                writer.add_scalar('Speed', tkps, global_num_tokens_consumed)
                writer.add_scalar('PeakMemoryUsage', peak_memory_usage, global_num_tokens_consumed)
                writer.flush()
            # reset
            t_start = time.time()
            statistics[:] = 0
        dt_acc += time.time() - t_stat_start
        dt_helpers.put(dt_acc)
        dt_total.put(time.time()-t_step_start)



if __name__ == "__main__":

    # set arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_ckpt_dir", type=str, required=True)
    parser.add_argument("--save_ckpt_dir", type=str, required=True)
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--tokenizer_model", type=str, required=True)
    parser.add_argument("--tensor_type", type=str, required=True)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, required=True)
    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--prefetch_factor", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=4*1024)
    parser.add_argument("--warmup_tokens_b", type=float, default=0.2)
    parser.add_argument("--first_cycle_tokens_b", type=float, default=15)
    parser.add_argument("--grad_clip", type=float, default=1)
    parser.add_argument("--train_ddp", action="store_true", help="using pytorch ddp rather than ShardedDDP.")
    parser.add_argument("--train_sft", action="store_true", help="train with sft data.")
    parser.add_argument("--max_lr", type=float, default=5e-5) # 5e-5
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--n_offload_skips", type=int, default=0)
    parser.add_argument("--activation", type=str, default="none",
                        required=False, choices=["gpu_shard", "cpu_offload", "checkpoint", "none"])
    parser.add_argument("--ckpt_save_tokens_b", type=float,
                        default=1, help="how many tokens (b) to save and evaluate checkpoint.")
    args = parser.parse_args()

    # run training
    train(args)
