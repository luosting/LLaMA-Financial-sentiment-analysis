# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import xformers.ops as xops

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from typing import List
from functools import partial
from fairscale.nn.checkpoint.checkpointing import checkpoint_wrapper


def ckpt(module, activation_data, activation, force_offload=False):
    if activation == "checkpoint":
        module = checkpoint_wrapper(module, activation_data, offload_to_cpu=False,
                                    activation_sharding=False)
    elif activation == "gpu_shard":
        module = checkpoint_wrapper(module, activation_data, offload_to_cpu=False,
                                    activation_sharding=True)
    elif activation == "cpu_offload" or force_offload:
        module = checkpoint_wrapper(module, activation_data, offload_to_cpu=True,
                                    activation_sharding=False)
    elif activation == "none":
        pass
    else:
        raise ValueError(activation)
    return module

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    activation: str = "Not init"
    n_offload_skips: int = 0 # 0 means offload all, positive number to skip some blocks from starting.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.dim // args.n_heads
        self.attn_bias = xops.fmha.attn_bias.LowerTriangularMask()
        # args.n_heads=64, model_parallel_size=8, args.dim=8192, self.head_dim=128


        self.wq = ColumnParallelLinear(
            args.dim, # hidden dim
            args.n_heads * self.head_dim, # split dim into multiple heads. heads are parallelized.
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim, # hidden dim
            self.n_kv_heads * self.head_dim, # 
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, # 64*128
            args.dim, # 8192
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        """
        if args.max_accumulated_chunks > 1:
            self.cache_k = torch.zeros(
                (
                    1,
                    args.max_seq_len * args.max_accumulated_chunks,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    1,
                    args.max_seq_len * args.max_accumulated_chunks,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
        """

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        seq_len_list: List[int]
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if True:
            # GQA: https://facebookresearch.github.io/xformers/components/ops.html
            if True:
                output_list = []
                for k in range(bsz):
                    attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seq_len_list[k])
                    output = xops.fmha.memory_efficient_attention(
                        xq[k:k+1,:,:,:],
                        repeat_kv(xk[k:k+1,:,:,:], self.n_rep),
                        repeat_kv(xv[k:k+1,:,:,:], self.n_rep),
                        attn_bias=attn_bias)
                    output_list.append(output)
                output = torch.cat(output_list, dim=0).contiguous().view(bsz, seqlen, -1)
            else:
                output = xops.fmha.memory_efficient_attention(
                    xq,
                    repeat_kv(xk, self.n_rep),
                    repeat_kv(xv, self.n_rep),
                    attn_bias=self.attn_bias).contiguous().view(bsz, seqlen, -1)
            output = self.wo(output)
            return output
        """
        assert bsz == self.cache_k.shape[0], bsz
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        """
        keys = xk
        values = xv


        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # xq.shape=torch.Size([1, 8, 2518, 128]), keys.shape=torch.Size([1, 8, 2518, 128])
        # values.shape=torch.Size([1, 8, 2518, 128]), scores.shape=torch.Size([1, 8, 2518, 2518])
        # mask.shape=torch.Size([1, 1, 2518, 2518])
        # output.shape=(1, 2518, 8*128)
        # rets.shape = torch.Size([1, 2518, 8192])
        if mask is not None:
            """
            #Mask
            [[0, -inf, -inf],
             [0,    0, -inf],
             [0,    0,    0]]
            """
            # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores[:,:,:,start_pos:] = scores[:,:,:,start_pos:] + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output.shape=(1, 2518, 8*128) => Local
        # wo.shape=(64*128, 8192)
        # ret = output*wo => (1, 2518, 8192)
        output2 = self.wo(output)
        return output2


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.params = args
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.mask = torch.full(
            (1, 1, self.params.max_seq_len, self.params.max_seq_len),
            float("-inf"), device='cuda', dtype=torch.float)
        self.mask = torch.triu(self.mask).to(torch.bfloat16)
        # self.n_heads=64, self.dim=8192, self.head_dim=128, args.multiple_of=4096, args.ffn_dim_multiplier=1.3

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        seq_len_list: List[int]
    ):
        if self.freqs_cis.device != x.device:
            self.freqs_cis = self.freqs_cis.to(x.device)
        seqlen = x.shape[1]
        mask = self.mask
        if seqlen > 1:
            if self.mask.shape[2] != seqlen:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=x.device, dtype=torch.float
                )
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)
        freqs_cis = self.freqs_cis[:seqlen]
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask, seq_len_list)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.activation_data = []

        ckpt_wrap = partial(ckpt, 
                            activation_data=self.activation_data,
                            activation=params.activation)

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            if layer_id >= params.n_offload_skips:
                module = ckpt_wrap(TransformerBlock(layer_id, params))
            else:
                module = TransformerBlock(layer_id, params)
            self.layers.append(module)

        self.norm = ckpt_wrap(RMSNorm(params.dim, eps=params.norm_eps))
        # self.output = ColumnParallelLinear(
        #     params.dim,  # dim
        #     params.vocab_size, # 32000
        #     bias=False, init_method=lambda x: x
        # )

        self.output = ColumnParallelLinear(
            params.dim, # dim
            params.vocab_size, # 32000
            bias=False, init_method=lambda x: x
        )


    def forward(self, tokens: torch.Tensor, start_pos: int=0, seq_len_list=[]):
        self.activation_data.clear() # clear
        print(tokens.shape)
        _bsz, seqlen = tokens.shape # 1*seqlen
        h = self.tok_embeddings(tokens) # 1*token_size*dim
        assert seqlen > 0
        if len(seq_len_list) == 0: seq_len_list = [seqlen]
        #print(f"Before layer: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB")
        for k, layer in enumerate(self.layers):
            h = layer(h, start_pos, seq_len_list)
            #print(f"After layer-{k}: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB")
        h = self.norm(h) # # 1*seqlen*dim
        output = self.output(h)
        output = output[:, -1, :3]  # mod
        #output = output.float()
        #output = self.output(h)
        # h.shape=torch.Size([1, 1719, 8192]), output.shape=torch.Size([1, 1719, 32000])
        #print(f"After output: {torch.cuda.memory_allocated('cuda')/1024**2:.1f} MB")
        self.activation_data.append(None) # add terminator
        return output
