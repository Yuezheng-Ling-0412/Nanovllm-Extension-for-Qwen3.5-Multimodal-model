import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.utils.context import get_context

from fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

import triton
import triton.language as tl

def causal_conv1d_prefill(x, weight, conv_state):
    x_padded = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(x_padded[:, :, -(weight.size(-1) - 1):])
    out = F.conv1d(x_padded, weight, groups=x.size(1), padding=0)
    return F.silu(out)

@triton.jit
def _causal_conv1d_decode_kernel(
    x_ptr,              
    conv_state_ptr,     
    weight_ptr,         
    out_ptr,            
    new_state_ptr,      
    D,                  
    K: tl.constexpr,    
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)   
    b_idx = tl.program_id(1)   

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offs < D

    x = tl.load(x_ptr + b_idx * D + d_offs, mask=mask_d, other=0.0).to(tl.float32)

    out = tl.zeros([BLOCK_D], dtype=tl.float32)

    state_batch_offset = b_idx * D * (K-1)
    for i in range(K - 1):
        w_i = tl.load(weight_ptr + d_offs * K + i, mask=mask_d, other=0.0).to(tl.float32)
        state_val = tl.load(conv_state_ptr + state_batch_offset + d_offs * (K-1) + i, mask=mask_d, other=0.0).to(tl.float32)
        out += state_val * w_i

    w_last = tl.load(weight_ptr + d_offs * K + (K - 1), mask=mask_d, other=0.0).to(tl.float32)
    out += x * w_last

    out = out * tl.sigmoid(out)

    tl.store(out_ptr + b_idx * D + d_offs, out.to(tl.bfloat16), mask=mask_d)

    for i in range(K - 1):
        if i < K - 2:
            src_val = tl.load(conv_state_ptr + state_batch_offset + d_offs * (K-1) + (i + 1), mask=mask_d, other=0.0).to(tl.float32)
        else:  
            src_val = x
        tl.store(new_state_ptr + state_batch_offset + d_offs * (K-1) + i, src_val.to(tl.bfloat16), mask=mask_d)

def causal_conv1d_decode_triton(
    x: torch.Tensor,             # [B, D, 1]
    weight: torch.Tensor,        # [D, 1, K]
    conv_state: torch.Tensor,    # [B, D, K-1]
) -> tuple[torch.Tensor, torch.Tensor]:
    B, D, _ = x.shape
    K = weight.shape[-1]
    assert weight.shape[0] == D and weight.shape[1] == 1
    assert conv_state.shape == (B, D, K-1), f"Expected conv_state shape ({B}, {D}, {K-1}), got {conv_state.shape}"

    x = x.contiguous()
    conv_state = conv_state.contiguous()
    weight_flat = weight.squeeze(1).contiguous()  # [D, K]

    out = torch.empty_like(x)
    new_state = torch.empty_like(conv_state)

    BLOCK_D = 128 
    grid = (triton.cdiv(D, BLOCK_D), B)

    _causal_conv1d_decode_kernel[grid](
        x, conv_state, weight_flat,
        out, new_state,
        D=D, K=K, BLOCK_D=BLOCK_D,
    )
    return out, new_state

class GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_v_heads = config.linear_num_value_heads
        self.total_num_k_heads = config.linear_num_key_heads
        assert self.total_num_v_heads % self.tp_size == 0
        assert self.total_num_k_heads % self.tp_size == 0
        self.num_v_heads = self.total_num_v_heads // self.tp_size
        self.num_k_heads = self.total_num_k_heads // self.tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.total_key_dim = self.head_k_dim * self.total_num_k_heads
        self.total_value_dim = self.head_v_dim * self.total_num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim 
        self.gqa_ratio = self.num_v_heads // self.num_k_heads  
        self.layer_idx = layer_idx

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_qkv.weight.weight_loader = self.qkv_weight_loader
        self.in_proj_z = ColumnParallelLinear(self.hidden_size, self.total_value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self.qkv_weight_loader
        
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log.weight_loader = self.head_weight_loader
        self.dt_bias.weight_loader = self.head_weight_loader
        
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.norm.weight = nn.Parameter(self.norm.weight.data.to(torch.float32))
        
        self.out_proj = RowParallelLinear(self.total_value_dim, self.hidden_size, bias=False)

        self.conv_states: torch.Tensor = torch.tensor([])   # (1, self.conv_dim, self.conv_kernel_size - 1)
        self.recurrent_states: torch.Tensor = torch.tensor([])  # (Dk, Dv)

    def qkv_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ):
        q_start = self.tp_rank * self.key_dim
        k_start = self.total_key_dim + self.tp_rank * self.key_dim
        v_start = 2 * self.total_key_dim + self.tp_rank * self.value_dim
        q = loaded_weight.narrow(0, q_start, self.key_dim)
        k = loaded_weight.narrow(0, k_start, self.key_dim)
        v = loaded_weight.narrow(0, v_start, self.value_dim)
        param.data.copy_(torch.cat([q, k, v], dim=0))

    def head_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ):
        start = self.tp_rank * self.num_v_heads
        param.data.copy_(loaded_weight.narrow(0, start, self.num_v_heads))

    def _forward_prefill(self, hidden_states):
        context = get_context()
        cu_seqlens = context.cu_seqlens_q 
        state_indices = context.state_indices
        num_seqs = cu_seqlens.size(0) - 1
        warmup = state_indices is None or self.conv_states.numel() == 0

        mixed_qkv = self.in_proj_qkv(hidden_states)   # [total_tokens, D]
        z = self.in_proj_z(hidden_states)             # [total_tokens, Dv]
        b = self.in_proj_b(hidden_states)             # [total_tokens, Hv]
        a = self.in_proj_a(hidden_states)             # [total_tokens, Hv]

        conv_weight = self.conv1d.weight
        conv_outputs = []
        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()

            x_chunk = mixed_qkv[start:end].unsqueeze(0).transpose(1, 2)  # [1, D, L]

            if warmup:
                conv_state = torch.zeros(1, self.conv_dim, self.conv_kernel_size - 1,
                                         dtype=x_chunk.dtype, device=x_chunk.device)
            else:
                si = state_indices[i].item()
                conv_state = self.conv_states[si:si+1] 

            out_conv = causal_conv1d_prefill(x_chunk, conv_weight, conv_state)
            out_conv = out_conv.squeeze(0).transpose(0, 1)   # [L, D]
            conv_outputs.append(out_conv)

        mixed_qkv = torch.cat(conv_outputs, dim=0)

        q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, -1, self.num_k_heads, self.head_k_dim)   # [1, total_tokens, Hk, Dk]
        k = k.reshape(1, -1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, -1, self.num_v_heads, self.head_v_dim)   # [1, total_tokens, Hv, Dv]

        a = a.unsqueeze(0)    # [1, total_tokens, Hv]
        b = b.unsqueeze(0)

        if warmup:
            initial_state = None
        else:
            initial_state = self.recurrent_states[state_indices]   # [num_seqs, Hv, Dk, Dv]

        out, new_states = chunk_gated_delta_rule(
            q, k, v,
            g=a,                                   
            beta=b,                                
            scale=self.head_k_dim ** -0.5,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,          
            use_gate_in_kernel=True,               
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_beta_sigmoid_in_kernel=True,       
            cu_seqlens=cu_seqlens,
        )  

        if not warmup:
            self.recurrent_states[state_indices] = new_states

        out_flat = out.reshape(-1, self.head_v_dim)        # [total_tokens, Dv]
        normed = self.norm(out_flat, z)                    # [total_tokens, Dv]
        output = self.out_proj(normed.reshape(1, -1, self.value_dim))  # [1, total_tokens, out_dim]
        return output.squeeze(0)                           # [total_tokens, out_dim]

    def _forward_decode(self, hidden_states):
        context = get_context()
        state_indices = context.state_indices
        B = hidden_states.size(0)

        x = hidden_states.unsqueeze(1)                          # [B, 1, H]
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)         # [B, D, 1]
        z = self.in_proj_z(x)                                   # [B, 1, Dv]
        b = self.in_proj_b(x)                                   # [B, 1, Hv]
        a = self.in_proj_a(x)                                   # [B, 1, Hv]

        conv_state = self.conv_states[state_indices]
        mixed_qkv, new_conv_state = causal_conv1d_decode_triton(mixed_qkv, self.conv1d.weight, conv_state)
        self.conv_states[state_indices] = new_conv_state

        mixed_qkv = mixed_qkv.transpose(1, 2)                   # [B, 1, D]
        q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim)  # [B, 1, Hk, Dk]
        k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim)  # [B, 1, Hv, Dv]

        rec_state = self.recurrent_states[state_indices]        # [B, Hv, Dk, Dv]

        out, new_state = fused_recurrent_gated_delta_rule(
            q, k, v,
            g=a,                                   
            beta=b,                                
            scale=self.head_k_dim ** -0.5,
            initial_state=rec_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,          
            use_gate_in_kernel=True,               
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_beta_sigmoid_in_kernel=True,       
        )  # out: [B, 1, Hv, Dv], new_state: [B, Hv, Dk, Dv]

        self.recurrent_states[state_indices] = new_state

        z_flat = z.reshape(-1, self.head_v_dim)     # [B, Dv]
        out_flat = out.reshape(-1, self.head_v_dim) # [B, Dv]
        normed = self.norm(out_flat, z_flat)        # [B, Dv]
        output = self.out_proj(normed.reshape(B, 1, self.value_dim)).squeeze(1)  # [B, out_dim]
        return output

    def forward(self, hidden_states):
        context = get_context()
        if context.is_prefill:
            return self._forward_prefill(hidden_states)
        else:
            return self._forward_decode(hidden_states)
