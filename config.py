from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    model_type: str = "transformer"  # "transformer" (ALiBi Seq2Seq)
    d_model: int = 256  # Hidden dimension 크기

    # Transformer 관련 설정
    nhead: int = 8  # Multi-head Attention의 head 수
    num_encoder_layers: int = 6  # 인코더 레이어 수
    num_decoder_layers: int = 6  # 디코더 레이어 수
    dim_feedforward: int = 1024  # Feedforward 네트워크의 hidden 차원
    dropout: float = 0.1  # 드롭아웃 비율


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3
    valid_every: int = 50
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 4
    save_best_path: Optional[str] = None
    stage1_epochs: int = 4
    stage2_epochs: int = 0
    lambda_aux: float = 0.2
    lambda_contrast: float = 0.1


