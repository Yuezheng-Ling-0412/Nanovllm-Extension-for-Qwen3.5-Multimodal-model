from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_partial = rotary_dim < head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_partial:
            q_rot = query[..., :self.rotary_dim]
            q_pass = query[..., self.rotary_dim:]
            k_rot = key[..., :self.rotary_dim]
            k_pass = key[..., self.rotary_dim:]
            q_rot = apply_rotary_emb(q_rot, cos, sin)
            k_rot = apply_rotary_emb(k_rot, cos, sin)
            query = torch.cat((q_rot, q_pass), dim=-1)
            key = torch.cat((k_rot, k_pass), dim=-1)
        else:
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)
        return query, key


class InterleavedMRoPE(nn.Module):
    def __init__(
        self,
        head_size: int,
        partial_rotary_factor: float,
        mrope_section: list[int],
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = int(head_size * partial_rotary_factor)
        self.is_partial = self.rotary_dim < head_size
        self.mrope_section = mrope_section 
        half_dim = self.rotary_dim // 2  
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply_interleaved_mrope(self, freqs):
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self,
        positions: torch.Tensor, 
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if positions.ndim == 1:
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
        else:
            positions_3d = positions
            inv_freq = self.inv_freq.float()
            freqs = torch.einsum("dn,h->dnh", positions_3d.float(), inv_freq)

            merged = self._apply_interleaved_mrope(freqs)

            cos = merged.cos().unsqueeze(1)
            sin = merged.sin().unsqueeze(1)

        if self.is_partial:
            q_rot = query[..., :self.rotary_dim]
            q_pass = query[..., self.rotary_dim:]
            k_rot = key[..., :self.rotary_dim]
            k_pass = key[..., self.rotary_dim:]
            q_rot = apply_rotary_emb(q_rot, cos, sin)
            k_rot = apply_rotary_emb(k_rot, cos, sin)
            query = torch.cat((q_rot, q_pass), dim=-1)
            key = torch.cat((k_rot, k_pass), dim=-1)
        else:
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=4)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
