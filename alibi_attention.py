"""
Custom Multi-Head Attention with ALiBi (Attention with Linear Biases)

Hugging Face Transformers의 BLOOM 모델 구현을 참고하여 작성
Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def build_alibi_tensor(num_heads: int, max_seq_len: int, device: torch.device) -> torch.Tensor:
    """
    ALiBi bias tensor 생성 (BLOOM 구현 기반)
    
    Args:
        num_heads: Attention head 수
        max_seq_len: 최대 시퀀스 길이
        device: 텐서가 위치할 디바이스
        
    Returns:
        alibi_bias: [1, num_heads, max_seq_len, max_seq_len]
    """
    # Closest power of 2
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    
    # Base slopes (기하급수적 감소)
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, 1 + closest_power_of_2, device=device)
    slopes = torch.pow(base, powers)
    
    # num_heads가 2의 거듭제곱이 아닌 경우
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = num_heads - closest_power_of_2
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    
    # 상대적 위치 행렬 생성
    # arange_tensor: [max_seq_len]
    arange_tensor = torch.arange(max_seq_len, device=device)
    
    # alibi: [max_seq_len, max_seq_len]
    # alibi[i, j] = j - i (상대적 거리)
    alibi = arange_tensor[None, :] - arange_tensor[:, None]
    alibi = alibi.abs().mul(-1)  # 음수로 변환
    
    # Head별로 slope 적용: [num_heads, max_seq_len, max_seq_len]
    alibi = alibi.unsqueeze(0) * slopes.view(num_heads, 1, 1)
    
    # Batch 차원 추가: [1, num_heads, max_seq_len, max_seq_len]
    alibi = alibi.unsqueeze(0)
    
    return alibi


class ALiBiMultiHeadAttention(nn.Module):
    """
    ALiBi를 사용하는 Multi-Head Attention
    
    BLOOM의 구현을 참고하여 Encoder-Decoder Transformer에 맞게 수정
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 1024,
    ):
        """
        Args:
            d_model: 모델의 차원
            num_heads: Attention head 수
            dropout: Dropout 비율
            bias: Linear layer에 bias 사용 여부
            max_seq_len: 최대 시퀀스 길이 (ALiBi 캐싱용)
        """
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Q, K, V projection
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slopes 저장 (학습되지 않음)
        self.register_buffer(
            "alibi_slopes",
            self._get_alibi_slopes(num_heads)
        )
        
        self.max_seq_len = max_seq_len
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """
        ALiBi slopes 계산 (BLOOM 방식)
        
        Args:
            num_heads: Head 수
            
        Returns:
            slopes: [num_heads]
        """
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = torch.arange(1, 1 + closest_power_of_2)
        slopes = torch.pow(base, powers)
        
        if closest_power_of_2 != num_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            num_remaining_heads = num_heads - closest_power_of_2
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        
        return slopes
    
    def _get_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        현재 시퀀스 길이에 맞는 ALiBi bias 생성
        
        Args:
            seq_len_q: Query 시퀀스 길이
            seq_len_k: Key 시퀀스 길이
            device: 디바이스
            
        Returns:
            alibi_bias: [1, num_heads, seq_len_q, seq_len_k]
        """
        # 상대적 위치 계산
        # query: [0, 1, 2, ..., seq_len_q-1]
        # key: [0, 1, 2, ..., seq_len_k-1]
        query_pos = torch.arange(seq_len_q, device=device)
        key_pos = torch.arange(seq_len_k, device=device)
        
        # relative_pos: [seq_len_q, seq_len_k]
        # relative_pos[i, j] = j - i
        relative_pos = key_pos[None, :] - query_pos[:, None]
        relative_pos = relative_pos.abs().mul(-1).float()  # 음수로 변환
        
        # Head별로 slope 적용: [num_heads, seq_len_q, seq_len_k]
        alibi_bias = relative_pos.unsqueeze(0) * self.alibi_slopes.view(-1, 1, 1)
        
        # Batch 차원 추가: [1, num_heads, seq_len_q, seq_len_k]
        return alibi_bias.unsqueeze(0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ALiBi를 사용한 Multi-Head Attention
        
        Args:
            query: Query 텐서 [batch, seq_len_q, d_model]
            key: Key 텐서 [batch, seq_len_k, d_model]
            value: Value 텐서 [batch, seq_len_k, d_model]
            key_padding_mask: Key padding mask [batch, seq_len_k] (True는 무시)
            attn_mask: Attention mask [seq_len_q, seq_len_k] (True는 무시)
            need_weights: Attention weights 반환 여부
            
        Returns:
            output: [batch, seq_len_q, d_model]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_k] (optional)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.size(1)
        
        # Projections
        Q = self.q_proj(query)  # [batch, seq_len_q, d_model]
        K = self.k_proj(key)    # [batch, seq_len_k, d_model]
        V = self.v_proj(value)  # [batch, seq_len_k, d_model]
        
        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, num_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        # ✨ ALiBi bias 추가 (핵심!)
        alibi_bias = self._get_alibi_bias(seq_len_q, seq_len_k, query.device)
        attn_scores = attn_scores + alibi_bias
        
        # Key padding mask 적용
        if key_padding_mask is not None:
            # [batch, seq_len_k] -> [batch, 1, 1, seq_len_k]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Attention mask 적용 (causal mask 등)
        if attn_mask is not None:
            # [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum: [batch, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # Reshape back: [batch, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        else:
            return output, None


class ALiBiTransformerEncoderLayer(nn.Module):
    """
    ALiBi를 사용하는 Transformer Encoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Self-Attention with ALiBi
        self.self_attn = ALiBiMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, d_model]
            src_mask: [seq_len, seq_len]
            src_key_padding_mask: [batch, seq_len]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-Attention
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ALiBiTransformerDecoderLayer(nn.Module):
    """
    ALiBi를 사용하는 Transformer Decoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Self-Attention with ALiBi
        self.self_attn = ALiBiMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Cross-Attention with ALiBi
        self.cross_attn = ALiBiMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch, seq_len_tgt, d_model]
            memory: [batch, seq_len_src, d_model]
            tgt_mask: [seq_len_tgt, seq_len_tgt]
            memory_mask: [seq_len_tgt, seq_len_src]
            tgt_key_padding_mask: [batch, seq_len_tgt]
            memory_key_padding_mask: [batch, seq_len_src]
            
        Returns:
            output: [batch, seq_len_tgt, d_model]
        """
        # Self-Attention
        tgt2, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-Attention
        tgt2, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

