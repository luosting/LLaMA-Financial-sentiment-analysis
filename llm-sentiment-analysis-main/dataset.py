"""
Datasets from both further-pretraining and SFT.
"""
import random
import typing
import torch
import time
import struct
import glob
import tqdm
import os
import pickle
import queue
import sys
import threading
from llama.tokenizer import Tokenizer

class SFTData(torch.utils.data.Dataset):
    def __init__(
            self,
            seed:int,
            data_file:str, # a file containing a list of training instances.
            max_seq_len, # if a document exceeds this value, only up to the max_seq_len will be used.
            tokenizer,
            num_epochs, # how many epoches to iterate the data.
    ):
        # distributed
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
        assert WORLD_SIZE > -1
        self.dp_world_size = WORLD_SIZE
        self.dp_rank = torch.distributed.get_rank()
        # set params
        self.data_file = data_file
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        # load prompt template
        with open('prompt.txt') as fp:
            # news_title
            prompt_template = fp.read().strip()
        # load training instances
        labelmap = {'positive': 'A', 'negative': 'B', 'neutral': 'C'}
        B_INST, E_INST = "[INST]", "[/INST]"
        self.train_data = [] # (tokens, size_qa, size_q, size_a)
        with open(data_file) as fp:
            lines = fp.read().strip().split('\n')
            for l in lines:
                label, text = l.split('\t\t')
                answer = labelmap[label]
                question = prompt_template.format(news_title=text)
                token_q = tokenizer.encode(f'{B_INST} {question} {E_INST}', bos=True, eos=False)
                token_a = tokenizer.encode(f'{answer}', bos=False, eos=False)
                size_q = len(token_q)
                size_a = len(token_a)
                size_qa = size_q + size_a
                tokens = token_q + token_a
                assert size_qa <= self.max_seq_len, f'{size_qa=} {self.max_seq_len=}'
                # assert False, f'{size_q=} {size_a=}'
                self.train_data.append((tokens, size_qa, size_q, size_a))
        self.read_pos = 0
        # duplicates the data
        full_data = []
        for _ in range(num_epochs):
            full_data += self.train_data
        self.train_data = full_data
        # shuffle data.
        random.seed(seed)
        random.shuffle(self.train_data)

    def next(self):
        """
        Return the next item for training in distributed data parallel mode.
        Each entry has the following protocol: [x, y, seq_len], where x,y with shape
        (max_seq_len,). On termination, it raise StopIteration exception.
        """
        # check termination
        if self.read_pos >= len(self.train_data):
            raise StopIteration
        # prepare the full global batch across all data-parallel workers and select one according to my rank.
        global_batch = [None for _ in range(self.dp_world_size)]
        for k in range(self.dp_world_size):
            # for each rank: prepare an instance.
            num_to_fetch = self.max_seq_len
            X = []
            Y = []
            seq_len = []
            while num_to_fetch > 0:
                # construct X, and seq_len.
                if self.read_pos < len(self.train_data) and num_to_fetch >= self.train_data[self.read_pos][1]:
                    # fill X with one training entry.
                    tokens, size_qa, size_q, size_a = self.train_data[self.read_pos]
                    X += tokens
                    Y += [self.tokenizer.pad_id for _ in range(size_q-1)] + tokens[-size_a:] + [self.tokenizer.pad_id]
                    seq_len.append(size_qa)
                    num_to_fetch -= len(tokens)
                    self.read_pos += 1
                elif num_to_fetch > 0:
                    # cannot fit or data done: using pad for X.
                    X += [0 for _ in range(num_to_fetch)]
                    Y += [self.tokenizer.pad_id for _ in range(num_to_fetch)]
                    seq_len.append(num_to_fetch)
                    num_to_fetch = 0
            assert len(X) == len(Y)
            assert sum(seq_len) == len(X)
            assert len(X) == self.max_seq_len, f'{len(X)=} {self.max_seq_len=}'
            global_batch[k] = [X, Y, seq_len]
        if max(global_batch[self.dp_rank][1]) < 0:
            # no valid data.
            # then use the first rank.
            return global_batch[0]
        else:
            return global_batch[self.dp_rank]

class PretrainData(torch.utils.data.Dataset):
    """
    Distributed dataset for further-pretraining.
    """
    def __init__(
            self,
            seed:int,
            data_dir:str, # a folder containing *.tk for training tokens.
            max_seq_len, # if a document exceeds this value, only up to the max_seq_len will be used.
            pad_id, # token used for padding.
    ):
        # distributed
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
        assert WORLD_SIZE > -1
        self.dp_world_size = WORLD_SIZE
        self.dp_rank = torch.distributed.get_rank()
        # set params
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        # load token file list.
        self.all_tk_files = glob.glob(f'{data_dir}/*.tk')
        assert len(self.all_tk_files) > 0, data_dir
        self.pos_block = 0
        self.pos_item = 0
        self.block_tokens = []
        self.data_to_consume = None
        # shuffle data.
        random.seed(seed)
        random.shuffle(self.all_tk_files)

    def load_tokens(self):
        """
        read one block of tokens into self.block_tokens and reset counters.
        """
        self.pos_item = 0
        self.pos_block += 1
        if self.pos_block >= len(self.all_tk_files):
            raise StopIteration
        with open(self.all_tk_files[self.pos_block], 'rb') as fp:
            byte_data = fp.read()
            assert len(byte_data) % 4 == 0, len(byte_data)
            N = int(len(byte_data) / 4)
            self.block_tokens = struct.unpack(f'{N}i', byte_data)

    def next(self):
        """
        Return the next item for training in distributed data parallel mode.
        Each entry has the following protocol: [x, y, seq_len], where x,y with shape
        (max_seq_len,). On termination, it raise StopIteration exception (discard the last partial global batch).
        """
        # prepare the full global batch across all data-parallel workers and select one according to my rank.
        global_batch = [None for _ in range(self.dp_world_size)]
        for k in range(self.dp_world_size):
            # for each rank: prepare an instance.
            num_to_fetch = self.max_seq_len
            X = []
            seq_len = []
            while num_to_fetch > 0:
                # fetch one item from self.block_tokens
                if self.data_to_consume is None:
                    if self.pos_item >= len(self.block_tokens):
                        # block compltes, load a new block.
                        self.load_tokens()
                    num_tokens = self.block_tokens[self.pos_item]
                    beg = self.pos_item + 1 # beg pos of token data
                    end = beg + num_tokens  # end pos of token data
                    self.pos_item += (num_tokens + 1)
                    self.data_to_consume = self.block_tokens[beg:end]
                    if len(self.data_to_consume) > self.max_seq_len:
                        # truncate the data if it is too long.
                        self.data_to_consume = self.data_to_consume[:self.max_seq_len]
                # construct X, and seq_len.
                if num_to_fetch >= len(self.data_to_consume):
                    X += self.data_to_consume
                    num_to_fetch -= len(self.data_to_consume)
                    seq_len.append(len(self.data_to_consume))
                    self.data_to_consume = None # mark the fetched data as consumed.
                elif num_to_fetch > 0:
                    # cannot fit: using pad for X.
                    X += [self.pad_id for _ in range(num_to_fetch)]
                    seq_len.append(num_to_fetch)
                    break
            # construct Y according to the X.
            Y = X[1:] + [self.pad_id]
            for kk in range(len(X)):
                if X[kk] == self.pad_id: X[kk] = 0
            assert len(X) == len(Y)
            assert len(X) == self.max_seq_len
            assert sum(seq_len) == len(X)
            seq_len = [self.max_seq_len]
            global_batch[k] = [X, Y, seq_len]
        return global_batch[self.dp_rank]

class PrefetchDataLoader(threading.Thread):
    """
    Distributed data loader supporting prefetching.
    """
    def __init__(self, dataset, max_seq_len,
                 prefetch_factor=3,
                 micro_batch_size=1):
        super().__init__()
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        assert self.local_rank >= 0
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.micro_batch_size = micro_batch_size

    def run(self):
        while True:
            micro_batch_data = []
            stop_iter = False
            for batch_index in range(self.micro_batch_size):
                try:
                    data = self.dataset.next()
                except StopIteration:
                    # discard the last partial batch.
                    stop_iter = True
                    break
                assert len(data) == 3, len(data) # X, Y, seq_len
                assert len(data[0]) == self.max_seq_len, f'{len(data[0])} vs {self.max_seq_len}'
                assert len(data[1]) == self.max_seq_len, f'{len(data[1])} vs {self.max_seq_len}'
                micro_batch_data.append(data)
            # construct a micro batch
            if len(micro_batch_data) == self.micro_batch_size:
                batch = [None for _ in range(3)] # X, Y, seq_len
                batch[0] = torch.tensor([d[0] for d in micro_batch_data],
                                        dtype=torch.long,
                                        device=f'cuda:{self.local_rank}')
                batch[1] = torch.tensor([d[1] for d in micro_batch_data],
                                        dtype=torch.long,
                                        device=f'cuda:{self.local_rank}')
                batch[2] = [d[2] for d in micro_batch_data] # 2-d list.
                self.queue.put(batch)
            if stop_iter: break # data completes.
        self.queue.put(None)

    def __next__(self):
        data = self.queue.get()
        if data is None:
            raise StopIteration
        else:
            return data

def test_sft():
    # init dist.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    # env
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    assert local_rank > -1
    assert WORLD_SIZE > 0
    global_rank = torch.distributed.get_rank()

    # stdout/stderr
    if global_rank > 0:
        pass
        # disable logging on worker machines.
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # tokenizer
    tokenizer_path='checkpoints/llama2_7b_base/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)

    # dataset
    seed = 123
    max_seq_len = 4096
    data_file = 'data/train_data.txt'
    dataset = SFTData(
        seed,
        data_file,
        max_seq_len,
        tokenizer,
    )

    # fetch data
    prefetch_factor = 5
    micro_batch_size = 4
    data_loader = PrefetchDataLoader(dataset, max_seq_len, prefetch_factor, micro_batch_size)
    data_loader.start()

    # looping
    index = 0
    t_start = time.time()
    total_tokens = 0
    while True:
        try:
            x, y, seq_len = next(data_loader)
            assert len(x) == len(seq_len)
            assert len(x[0]) == sum(seq_len[0])
            # print(seq_len)
            dt = time.time() - t_start
            total_tokens += x.numel() * WORLD_SIZE
            tkps = total_tokens / dt
            print(f'\r{global_rank=} {index=} {x.shape=} tk/s = {tkps:,} {total_tokens=:,}', end='')
        except StopIteration:
            break
        index += 1
    print('\nTEST OK')

def test_pretrain():
    # init dist.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    # env
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    assert local_rank > -1
    assert WORLD_SIZE > 0
    global_rank = torch.distributed.get_rank()

    # stdout/stderr
    if global_rank > 0:
        pass
        # disable logging on worker machines.
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # tokenizer
    tokenizer_path='checkpoints/llama2_7b_base/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)

    # dataset
    seed = 123
    max_seq_len = 4096
    data_dir = 'data/reuters_tokens'
    dataset = PretrainData(
        seed,
        data_dir,
        max_seq_len,
        tokenizer.pad_id,
    )

    # fetch data
    prefetch_factor = 5
    micro_batch_size = 4
    data_loader = PrefetchDataLoader(dataset, max_seq_len, prefetch_factor, micro_batch_size)
    data_loader.start()

    # looping
    index = 0
    t_start = time.time()
    total_tokens = 0
    while True:
        try:
            x, y, seq_len = next(data_loader)
            assert len(x) == len(seq_len)
            assert len(x[0]) == sum(seq_len[0])
            # print(seq_len)
            dt = time.time() - t_start
            total_tokens += x.numel() * WORLD_SIZE
            tkps = total_tokens / dt
            if index % 100 == 0:
                print(f'\r{global_rank=} {index=} {x.shape=} tk/s = {tkps:,} {total_tokens=:,}', end='')
        except StopIteration:
            break
        index += 1
    print('\nTEST OK')


if __name__ == "__main__":
    test_sft()
