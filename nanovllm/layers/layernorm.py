import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_fused(
    X_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_y_row,
    N, eps, offset,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_start_ptr = X_ptr + row_idx * stride_x_row
    y_row_start_ptr = Y_ptr + row_idx * stride_y_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    x = tl.load(x_row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    
    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    sum_sq = tl.sum(x_sq, axis=0)
    rsqrt = tl.math.rsqrt((sum_sq / N) + eps)
    
    y = x_f32 * rsqrt * (w + offset)
    y_out = y.to(x.dtype)
    tl.store(y_row_start_ptr + col_offsets, y_out, mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, offset: float = 0.0):
    x = x.contiguous()
    y = torch.empty_like(x)
    
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _rmsnorm_fwd_fused[grid](
        x, y, weight,
        N, N,
        N, eps, offset,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

class GemmaRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    # @torch.compile(dynamic=True)
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        if residual is not None:
            x = x + residual
            residual = x
        
        x = triton_rmsnorm(x, self.weight, 1e-6, 1.0)

        return (
            x if residual is None else (x, residual)
        )


@triton.jit
def _rmsnorm_gated_fwd_fused(
    X_ptr, Y_ptr, W_ptr, G_ptr,
    stride_x_row, stride_y_row, stride_g_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_start_ptr = X_ptr + row_idx * stride_x_row
    y_row_start_ptr = Y_ptr + row_idx * stride_y_row
    g_row_start_ptr = G_ptr + row_idx * stride_g_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    x = tl.load(x_row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    g = tl.load(g_row_start_ptr + col_offsets, mask=mask, other=0.0)
    
    x_f32 = x.to(tl.float32)
    g_f32 = g.to(tl.float32)
    x_sq = x_f32 * x_f32
    sum_sq = tl.sum(x_sq, axis=0)
    rsqrt = tl.math.rsqrt((sum_sq / N) + eps)
    
    y = x_f32 * rsqrt * w
    g_silu = g_f32 * tl.sigmoid(g_f32)
    y = y * g_silu
    y_out = y.to(x.dtype)
    tl.store(y_row_start_ptr + col_offsets, y_out, mask=mask)

def triton_rmsnorm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-6):
    x = x.contiguous()
    gate = gate.contiguous()
    y = torch.empty_like(x)
    
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _rmsnorm_gated_fwd_fused[grid](
        x, y, weight, gate,
        N, N, N,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

class RMSNormGated(nn.Module):
    """Gated RMSNorm: output = RMSNorm(x) * SiLU(gate)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    # @torch.compile(dynamic=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        
        return triton_rmsnorm_gated(hidden_states, self.weight, gate, self.eps)
