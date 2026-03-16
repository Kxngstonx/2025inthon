"""
InThon 2025 Datathon - Transformer Seq2Seq with ALiBi

이 파일은 ALiBi (Attention with Linear Biases)를 사용하는 
Encoder-Decoder Transformer 모델을 구현합니다.

주요 구성 요소:
- CharTokenizer: 문자 단위 토크나이저
- TransformerSeq2SeqWithALiBi: ALiBi를 사용하는 Seq2Seq 모델
- Model: 제출용 BaseModel 래퍼
"""

from __future__ import annotations
from typing import List, Dict, Optional
import re

import torch
import torch.nn as nn

from do_not_edit.model_template import BaseModel

# ============================================================
# 토크나이저
# ============================================================

# 특수 토큰
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

# 입력/출력 문자 집합
INPUT_CHARS = list("0123456789+-*/()")
OUTPUT_CHARS = list("0123456789-")  # 마이너스 기호 포함


class CharTokenizer:
    """문자 단위 토크나이저"""
    
    def __init__(self, chars: List[str], add_special: bool):
        vocab = list(chars)
        self.pad = PAD if add_special else None
        self.bos = BOS if add_special else None
        self.eos = EOS if add_special else None
        
        if add_special:
            vocab = [PAD, BOS, EOS] + vocab
        
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str, add_bos_eos: bool) -> List[int]:
        """문자열을 정수 인덱스 리스트로 변환"""
        ids = []
        if add_bos_eos and self.bos is not None:
            ids.append(self.stoi[self.bos])
        
        for ch in s:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        
        if add_bos_eos and self.eos is not None:
            ids.append(self.stoi[self.eos])
        
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad] if self.pad is not None else 0

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos] if self.bos is not None else 1

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos] if self.eos is not None else 2


# ============================================================
# 배치 처리 유틸리티
# ============================================================

def _pad(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    """시퀀스 리스트를 동일한 길이로 패딩"""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [pad_id] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)


def tokenize_batch(batch: Dict[str, List[str]], 
                   input_tokenizer: CharTokenizer, 
                   output_tokenizer: CharTokenizer) -> Dict[str, torch.Tensor]:
    """배치를 토큰화하여 텐서로 변환"""
    input_texts = batch["input_text"]
    target_texts = batch["target_text"]
    
    # 입력 인코딩 (BOS/EOS 없음)
    src_ids = [input_tokenizer.encode(text, add_bos_eos=False) for text in input_texts]
    src = _pad(src_ids, input_tokenizer.pad_id)
    
    # 타겟 인코딩 (BOS/EOS 추가)
    tgt_ids = [output_tokenizer.encode(text, add_bos_eos=True) for text in target_texts]
    tgt = _pad(tgt_ids, output_tokenizer.pad_id)
    
    # 디코더 입력과 출력 분리
    tgt_inp = tgt[:, :-1]  # BOS ~ 마지막 이전
    tgt_out = tgt[:, 1:]   # 첫 번째 이후 ~ EOS
    
    return {
        "src": src,
        "tgt_inp": tgt_inp,
        "tgt_out": tgt_out,
    }


# ============================================================
# TransformerSeq2SeqWithALiBi 모델
# ============================================================

class TransformerSeq2SeqWithALiBi(nn.Module):
    """
    ALiBi (Attention with Linear Biases)를 사용하는 Encoder-Decoder Transformer
    
    Hugging Face BLOOM 구현을 참고하여 완전한 ALiBi를 구현했습니다.
    Positional Encoding 대신 Attention Score에 선형 bias를 추가합니다.
    """
    
    def __init__(
        self,
        in_vocab: int,
        out_vocab: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 임베딩 레이어
        self.src_embedding = nn.Embedding(in_vocab, d_model)
        self.tgt_embedding = nn.Embedding(out_vocab, d_model)
        
        # ALiBi는 Positional Encoding을 사용하지 않음
        # 대신 dropout만 적용
        self.dropout = nn.Dropout(p=dropout)
        
        # Custom ALiBi Encoder/Decoder Layers
        from alibi_attention import ALiBiTransformerEncoderLayer, ALiBiTransformerDecoderLayer
        
        self.encoder_layers = nn.ModuleList([
            ALiBiTransformerEncoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            ALiBiTransformerDecoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])
        
        # 출력 레이어 (체크포인트 호환성을 위해 out_proj로 명명)
        self.out_proj = nn.Linear(d_model, out_vocab)
        
        # 파라미터 초기화
        self._init_parameters()
    
    def _init_parameters(self):
        """파라미터 초기화"""
        import math
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(p, gain=1.0 / math.sqrt(2))
    
    def _create_padding_mask(self, src: torch.Tensor, pad_id: int) -> torch.Tensor:
        """패딩 마스크 생성 (True = 패딩, False = 실제 토큰)"""
        return src == pad_id
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask 생성 (디코더용)"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def encode_src(self, src: torch.Tensor, src_pad_id: int) -> torch.Tensor:
        """
        Encode source sequence and return encoder states.
        Returns:
            [batch_size, src_len, d_model]
        """
        src_emb = self.dropout(self.src_embedding(src) * (self.d_model ** 0.5))
        src_padding_mask = self._create_padding_mask(src, src_pad_id)
        memory = src_emb.transpose(0, 1)
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(src=memory, src_key_padding_mask=src_padding_mask)
        return memory.transpose(0, 1)

    def mean_pool_encoder(self, encoder_states: torch.Tensor, src: torch.Tensor, src_pad_id: int) -> torch.Tensor:
        """
        Mean pool encoder states with non-pad mask.
        Args:
            encoder_states: [B, S, D]
            src: [B, S]
        Returns:
            pooled: [B, D]
        """
        mask = (src != src_pad_id).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (encoder_states * mask).sum(dim=1) / denom
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_pad_id: int) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src: 소스 시퀀스 [batch_size, src_len]
            tgt: 타겟 시퀀스 (디코더 입력) [batch_size, tgt_len]
            src_pad_id: 소스 패딩 토큰 ID
            
        Returns:
            logits [batch_size, tgt_len, out_vocab]
        """
        device = src.device
        
        # 임베딩 + 드롭아웃 (ALiBi는 PE 없음)
        tgt_emb = self.dropout(self.tgt_embedding(tgt) * (self.d_model ** 0.5))
        
        # 마스크 생성
        src_padding_mask = self._create_padding_mask(src, src_pad_id)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), device)
        
        # Encoder [B, S, D] -> [S, B, D]
        memory = self.encode_src(src, src_pad_id).transpose(0, 1)
        
        # Decoder
        output = tgt_emb.transpose(0, 1)
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(
                tgt=output,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
        
        # [seq, batch, d_model] → [batch, seq, d_model]
        output = output.transpose(0, 1)
        
        # 출력 레이어
        logits = self.out_proj(output)
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
        src_pad_id: int,
    ) -> torch.Tensor:
        """
        Greedy Decoding으로 시퀀스 생성
        
        Args:
            src: 소스 시퀀스 [batch_size, src_len]
            max_len: 최대 생성 길이
            bos_id: 시작 토큰 ID
            eos_id: 종료 토큰 ID
            src_pad_id: 소스 패딩 토큰 ID
            
        Returns:
            생성된 시퀀스 [batch_size, gen_len]
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # 소스 임베딩
        src_padding_mask = self._create_padding_mask(src, src_pad_id)
        
        # Encoder forward
        memory = self.encode_src(src, src_pad_id).transpose(0, 1)
        
        # 디코더 입력 초기화 (BOS 토큰으로 시작)
        decoder_input = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        # Auto-regressive 생성
        for _ in range(max_len):
            tgt_len = decoder_input.size(1)
            
            # 디코더 임베딩
            tgt_emb = self.dropout(self.tgt_embedding(decoder_input) * (self.d_model ** 0.5))
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
            
            # Decoder forward
            decoder_output = tgt_emb.transpose(0, 1)
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(
                    tgt=decoder_output,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
            
            decoder_output = decoder_output.transpose(0, 1)
            
            # 마지막 토큰의 logits
            logits = self.out_proj(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 생성된 토큰 추가
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # EOS 토큰이 모든 배치에서 생성되면 중단
            if (next_token == eos_id).all():
                break
        
        # BOS 토큰 제거하고 반환
        return decoder_input[:, 1:]


# ============================================================
# 제출용 Model 클래스
# ============================================================

class Model(BaseModel):
    """
    InThon Datathon 제출용 Model 클래스
    
    TransformerSeq2SeqWithALiBi 모델을 사용합니다.
    """
    
    def __init__(self, model_path: str = "best_model.pt") -> None:
        """
        모델 초기화
        
        Args:
            model_path: 모델 체크포인트 경로 (상대 경로, 기본값: "best_model.pt")
        """
        super().__init__()
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 설정 로드
        tokenizer_config = checkpoint.get("tokenizer_config")
        model_config = checkpoint.get("model_config")
        
        if tokenizer_config is None or model_config is None:
            raise ValueError("체크포인트에 'tokenizer_config' 또는 'model_config'가 없습니다.")
        
        # 모델 타입 저장
        self.model_type = model_config.get("model_type", "transformer")
        
        # Reversed sequence training 여부
        self.use_reversed = checkpoint.get("use_reversed", False)
        
        # 토크나이저 초기화
        input_chars = tokenizer_config.get("input_chars", INPUT_CHARS)
        output_chars = tokenizer_config.get("output_chars", OUTPUT_CHARS)
        add_special = tokenizer_config.get("add_special", True)
        
        self.input_tokenizer = CharTokenizer(input_chars, add_special=add_special)
        self.output_tokenizer = CharTokenizer(output_chars, add_special=add_special)
        
        # 모델 생성
        self.model = TransformerSeq2SeqWithALiBi(
            in_vocab=self.input_tokenizer.vocab_size,
            out_vocab=self.output_tokenizer.vocab_size,
            **{k: v for k, v in model_config.items() if k != "model_type"}
        ).to(self.device)
        
        # 가중치 로드
        model_state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(model_state)
        
        # 평가 모드
        self.model.eval()
        
        # 최대 생성 길이
        self.max_len = 50
    
    def predict(self, input_text: str) -> str:
        """
        입력 수식을 받아 계산 결과를 반환
        
        Args:
            input_text: 계산할 수식 문자열 (예: "12+34", "5*6")
            
        Returns:
            계산 결과 문자열 (예: "46", "30")
        """
        # 입력 타입 검증
        if not isinstance(input_text, str):
            input_text = str(input_text)
        
        # Reversed sequence training 적용
        input_text_processed = input_text
        if self.use_reversed:
            input_text_processed = self._reverse_arithmetic_numbers(input_text)
        
        # 토큰화
        batch = {"input_text": [input_text_processed], "target_text": ["0"]}
        batch_tensors = tokenize_batch(batch, self.input_tokenizer, self.output_tokenizer)
        src = batch_tensors["src"].to(self.device)
        
        # 추론
        with torch.no_grad():
            gens = self.model.generate(
                src=src,
                max_len=self.max_len,
                bos_id=self.output_tokenizer.bos_id,
                eos_id=self.output_tokenizer.eos_id,
                src_pad_id=self.input_tokenizer.pad_id,
            )
        
        # 생성된 토큰을 문자열로 변환
        seq_chars: List[str] = []
        for t in gens[0].tolist():
            idx = int(t)
            if idx == self.output_tokenizer.eos_id:
                break
            if idx in self.output_tokenizer.itos:
                ch = self.output_tokenizer.itos[idx]
                # 숫자와 마이너스 기호 추출
                if ch.isdigit() or (ch == '-' and not seq_chars):
                    seq_chars.append(ch)
        
        pred = "".join(seq_chars)
        
        # Reversed sequence training이었다면 결과를 다시 뒤집기
        if self.use_reversed:
            pred = self._reverse_arithmetic_numbers(pred)
        
        # 빈 문자열이거나 '-'만 있으면 "0" 반환
        if pred == "" or pred == "-":
            return "0"
        
        return pred
    
    def _reverse_arithmetic_numbers(self, text: str) -> str:
        """산술식의 숫자만 뒤집기 (연산자는 유지)"""
        def reverse_number(match):
            return match.group()[::-1]
        return re.sub(r'\d+', reverse_number, text)
