"""
Encoder-Decoder Transformer 모델 학습 스크립트

기존 train.py를 기반으로 TransformerSeq2Seq 모델을 사용하도록 수정한 버전입니다.
Weights & Biases (wandb) 통합으로 학습 과정을 추적할 수 있습니다.
"""

from __future__ import annotations
from typing import List, Any, Tuple, Dict, Optional
from config import TrainConfig, ModelConfig, TokenizerConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
import time
import sys
import math

# Wandb import (선택적)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb가 설치되지 않았습니다. 'pip install wandb'로 설치하세요.")

from dataloader import (
    ArithmeticDataset,
    get_dataloader,
)
from do_not_edit.metric import compute_metrics
from augmentation import AugmentConfig

from model import (
    TransformerSeq2SeqWithALiBi,  # ALiBi를 사용하는 Transformer 모델
    CharTokenizer,
    tokenize_batch,
    _pad,
    INPUT_CHARS,
    OUTPUT_CHARS,
)


# ======================================================================================
# 0. Reversed Sequence Helper Functions
# ======================================================================================
def reverse_arithmetic_numbers(text: str) -> str:
    """
    산술 식에서 숫자만 개별적으로 뒤집습니다.
    인간의 계산 방식(오른쪽에서 왼쪽)을 모방하기 위함입니다.
    
    예시:
    - "123+456" → "321+654"
    - "789-12" → "987-21"  
    - "99*7" → "99*7"
    - "579" → "975"
    
    Args:
        text: 입력 문자열 (산술 식 또는 숫자)
        
    Returns:
        숫자만 뒤집힌 문자열
    """
    import re
    
    def reverse_number(match):
        return match.group()[::-1]
    
    # 연속된 숫자 패턴을 찾아서 각각 뒤집기
    result = re.sub(r'\d+', reverse_number, text)
    return result


def _bt_get(batch_tensors: Any, key: str) -> torch.Tensor:
    if isinstance(batch_tensors, dict):
        return batch_tensors[key]
    return getattr(batch_tensors, key)


def _flatten_pair_lists(batch: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in batch.get(key, []):
        if isinstance(item, list):
            out.extend(item)
    return out


def _tokenize_inputs(input_texts: List[str], tokenizer: CharTokenizer, device: torch.device) -> torch.Tensor:
    src_ids = [tokenizer.encode(text, add_bos_eos=False) for text in input_texts]
    src = _pad(src_ids, tokenizer.pad_id)
    return src.to(device)


def _compute_symmetric_kl_for_aux(
    model: nn.Module,
    input_tokenizer: CharTokenizer,
    output_tokenizer: CharTokenizer,
    device: torch.device,
    aux_pairs: List[Dict[str, Any]],
    use_reversed: bool,
) -> torch.Tensor:
    if not aux_pairs:
        return torch.tensor(0.0, device=device)

    base_inputs = [p["base_expr"] for p in aux_pairs]
    aug_inputs = [p["aug_expr"] for p in aux_pairs]
    targets = [p["target"] for p in aux_pairs]

    if use_reversed:
        base_inputs = [reverse_arithmetic_numbers(t) for t in base_inputs]
        aug_inputs = [reverse_arithmetic_numbers(t) for t in aug_inputs]
        targets = [reverse_arithmetic_numbers(t) for t in targets]

    base_bt = tokenize_batch({"input_text": base_inputs, "target_text": targets}, input_tokenizer, output_tokenizer)
    aug_bt = tokenize_batch({"input_text": aug_inputs, "target_text": targets}, input_tokenizer, output_tokenizer)

    base_src = _bt_get(base_bt, "src").to(device)
    base_tgt_in = _bt_get(base_bt, "tgt_inp").to(device)
    base_tgt_out = _bt_get(base_bt, "tgt_out").to(device)

    aug_src = _bt_get(aug_bt, "src").to(device)
    aug_tgt_in = _bt_get(aug_bt, "tgt_inp").to(device)

    logits_base = model(base_src, base_tgt_in, input_tokenizer.pad_id)
    logits_aug = model(aug_src, aug_tgt_in, input_tokenizer.pad_id)

    logp_base = F.log_softmax(logits_base, dim=-1)
    p_base = F.softmax(logits_base, dim=-1)
    logp_aug = F.log_softmax(logits_aug, dim=-1)
    p_aug = F.softmax(logits_aug, dim=-1)

    kl_ba = F.kl_div(logp_base, p_aug, reduction="none").sum(dim=-1)
    kl_ab = F.kl_div(logp_aug, p_base, reduction="none").sum(dim=-1)
    mask = (base_tgt_out != output_tokenizer.pad_id).float()
    loss = ((kl_ba + kl_ab) * 0.5 * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def _compute_pair_contrastive_loss(
    model: nn.Module,
    input_tokenizer: CharTokenizer,
    device: torch.device,
    pairs: List[Dict[str, Any]],
    positive: bool,
    margin: float = 0.2,
) -> torch.Tensor:
    if not pairs:
        return torch.tensor(0.0, device=device)
    base_inputs = [p["base_expr"] for p in pairs]
    aug_inputs = [p["aug_expr"] for p in pairs]
    base_src = _tokenize_inputs(base_inputs, input_tokenizer, device)
    aug_src = _tokenize_inputs(aug_inputs, input_tokenizer, device)

    base_enc = model.encode_src(base_src, input_tokenizer.pad_id)
    aug_enc = model.encode_src(aug_src, input_tokenizer.pad_id)
    base_emb = model.mean_pool_encoder(base_enc, base_src, input_tokenizer.pad_id)
    aug_emb = model.mean_pool_encoder(aug_enc, aug_src, input_tokenizer.pad_id)

    cos = F.cosine_similarity(base_emb, aug_emb, dim=-1)
    if positive:
        return (1.0 - cos).mean()
    return F.relu(cos - margin).mean()


# ======================================================================================
# 1. Learning Rate Scheduler
# ======================================================================================
def get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    Warmup + Cosine Decay 스케줄러
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Warmup 단계 수 (선형 증가)
        total_steps: 전체 학습 단계 수
        min_lr_ratio: 최소 학습률 비율 (기본 lr의 10%)
    
    Returns:
        LambdaLR 스케줄러
    """
    def lr_lambda(current_step):
        # Warmup phase: 선형 증가 (0 → 1.0)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase: (1.0 → min_lr_ratio)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


# ======================================================================================
# 1. 학습 루프
# ======================================================================================
def train_loop(
    model: nn.Module,
    dataloader: DataLoader,
    input_tokenizer: CharTokenizer,
    output_tokenizer: CharTokenizer,
    device: torch.device,
    val_dataloader: DataLoader | None = None,
    *,
    train_config: TrainConfig,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    use_wandb: bool = False,
    early_stopping_patience: int = 10,
    enable_early_stopping: bool = True,
    use_reversed: bool = False,
    stage1_epochs: int = 0,
    stage2_epochs: int = 0,
    lambda_aux: float = 0.2,
    lambda_contrast: float = 0.1,
):
    """
    Transformer 모델 학습 루프
    
    Args:
        model: 학습할 모델 (TransformerSeq2Seq)
        dataloader: 학습 데이터 로더
        input_tokenizer: 입력 토크나이저
        output_tokenizer: 출력 토크나이저
        device: 학습 디바이스 (cpu/cuda)
        val_dataloader: 검증 데이터 로더
        train_config: 학습 설정
        model_config: 모델 설정
        tokenizer_config: 토크나이저 설정
        use_wandb: wandb 사용 여부
        early_stopping_patience: Early stopping patience (검증 성능 개선 없는 횟수)
        enable_early_stopping: Early stopping 활성화 여부
        use_reversed: Reversed sequence training 사용 여부 (1의 자리부터 예측)
    """
    model.to(device)
    
    # wandb로 모델 추적 (gradient, parameters)
    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=100)
    
    # 옵티마이저
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    
    # 총 step 수 계산 (스케줄러용)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * train_config.num_epochs
    if train_config.max_train_steps is not None:
        total_steps = min(total_steps, train_config.max_train_steps)
    
    # Learning rate 스케줄러: Warmup + Cosine Decay
    warmup_steps = min(500, total_steps // 20)  # 전체의 5% 또는 최대 500 step
    scheduler = get_warmup_cosine_schedule(
        optimizer=optim,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1  # 최소 학습률: 초기 lr의 10%
    )
    
    # Loss function: Label smoothing 추가
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_tokenizer.pad_id,
        label_smoothing=0.1  # Label smoothing으로 과신 방지
    )
    
    step = 0
    model.train()
    
    best_em = float("-inf")
    last_loss = 0.0
    start_time = time.time()
    last_print_time = time.time()
    
    # Early stopping 관련 변수
    patience_counter = 0  # 개선되지 않은 validation 횟수
    best_val_loss = float("inf")  # Best validation loss (낮을수록 좋음)
    early_stopped = False  # Early stopping 발동 여부
    
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"Total steps: {total_steps:,} | Steps per epoch: {steps_per_epoch:,} | Epochs: {train_config.num_epochs}")
    print(f"Learning rate: {train_config.lr:.6f} | Warmup steps: {warmup_steps}")
    print(f"Label smoothing: 0.1 | Gradient clipping: 0.5 | Max gen length: {train_config.max_gen_len}")
    if enable_early_stopping:
        print(f"Early stopping: Enabled (patience={early_stopping_patience})")
    else:
        print(f"Early stopping: Disabled")
    if use_reversed:
        print(f"🔄 Reversed Sequence Training: Enabled (숫자를 1의 자리부터 예측)")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    for epoch in range(train_config.num_epochs):
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
            break
        
        # Epoch 시작 메시지
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{train_config.num_epochs} Started")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        for batch in dataloader:
            # Reversed sequence training 적용
            if use_reversed:
                batch["input_text"] = [reverse_arithmetic_numbers(text) for text in batch["input_text"]]
                batch["target_text"] = [reverse_arithmetic_numbers(text) for text in batch["target_text"]]
            
            # 토크나이즈
            batch_tensors = tokenize_batch(batch, input_tokenizer, output_tokenizer)
            src = _bt_get(batch_tensors, "src").to(device)
            target_input = _bt_get(batch_tensors, "tgt_inp").to(device)
            target_output = _bt_get(batch_tensors, "tgt_out").to(device)
            
            # Forward
            logits = model(src, target_input, input_tokenizer.pad_id)
            
            # Loss 계산
            loss_ce = loss_fn(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                target_output.view(-1),             # (B*T,)
            )
            loss_aux = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)

            is_stage2 = stage2_epochs > 0 and epoch >= stage1_epochs
            if is_stage2:
                aux_pairs = _flatten_pair_lists(batch, "aux_pairs")
                ec_pairs = _flatten_pair_lists(batch, "ec_pairs")
                rc_pairs = _flatten_pair_lists(batch, "rc_pairs")
                boundary_extra = _flatten_pair_lists(batch, "boundary_extra")

                if aux_pairs:
                    loss_aux = _compute_symmetric_kl_for_aux(
                        model=model,
                        input_tokenizer=input_tokenizer,
                        output_tokenizer=output_tokenizer,
                        device=device,
                        aux_pairs=aux_pairs,
                        use_reversed=use_reversed,
                    )

                # Boundary samples are added as extra CE terms
                if boundary_extra:
                    boundary_inputs = [p["expr"] for p in boundary_extra]
                    boundary_targets = [p["target"] for p in boundary_extra]
                    if use_reversed:
                        boundary_inputs = [reverse_arithmetic_numbers(t) for t in boundary_inputs]
                        boundary_targets = [reverse_arithmetic_numbers(t) for t in boundary_targets]
                    boundary_bt = tokenize_batch(
                        {"input_text": boundary_inputs, "target_text": boundary_targets},
                        input_tokenizer,
                        output_tokenizer,
                    )
                    b_src = _bt_get(boundary_bt, "src").to(device)
                    b_tgt_in = _bt_get(boundary_bt, "tgt_inp").to(device)
                    b_tgt_out = _bt_get(boundary_bt, "tgt_out").to(device)
                    b_logits = model(b_src, b_tgt_in, input_tokenizer.pad_id)
                    loss_ce = loss_ce + loss_fn(
                        b_logits.view(-1, b_logits.size(-1)),
                        b_tgt_out.view(-1),
                    )

                ec_loss = _compute_pair_contrastive_loss(
                    model=model,
                    input_tokenizer=input_tokenizer,
                    device=device,
                    pairs=ec_pairs,
                    positive=True,
                    margin=0.2,
                )
                rc_loss = _compute_pair_contrastive_loss(
                    model=model,
                    input_tokenizer=input_tokenizer,
                    device=device,
                    pairs=rc_pairs,
                    positive=False,
                    margin=0.2,
                )
                loss_contrast = ec_loss + rc_loss

            loss = loss_ce + (lambda_aux * loss_aux) + (lambda_contrast * loss_contrast)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 1.0 → 0.5 (더 안정적)
            optim.step()
            scheduler.step()  # Learning rate 업데이트
            optim.zero_grad()
            
            step += 1
            last_loss = loss.item()
            
            # 진행 상황 출력 (100 step마다)
            current_time = time.time()
            if step % 100 == 0 or (current_time - last_print_time) >= 60:  # 100 step마다 또는 1분마다
                elapsed = current_time - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                remaining_steps = total_steps - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = int(eta_seconds // 3600)
                eta_mins = int((eta_seconds % 3600) // 60)
                
                progress_pct = (step / total_steps) * 100
                
                print(f"[Step {step:6d}/{total_steps}] "
                      f"Epoch {epoch+1}/{train_config.num_epochs} | "
                      f"Progress: {progress_pct:5.2f}% | "
                      f"Loss: {last_loss:.4f} | "
                      f"Speed: {steps_per_sec:.2f} step/s | "
                      f"ETA: {eta_hours}h {eta_mins}m")
                sys.stdout.flush()
                last_print_time = current_time
            
            # wandb에 학습 loss 및 현재 학습률 로깅
            if use_wandb and WANDB_AVAILABLE:
                current_lr = scheduler.get_last_lr()[0]
                rc_quality_items = [q for q in batch.get("rc_quality", []) if isinstance(q, dict) and q]
                rc_accept_rate = (
                    sum(float(q.get("accept_rate", 0.0)) for q in rc_quality_items) / len(rc_quality_items)
                    if rc_quality_items
                    else 0.0
                )
                rc_similarity_mean = (
                    sum(float(q.get("avg_similarity", 0.0)) for q in rc_quality_items) / len(rc_quality_items)
                    if rc_quality_items
                    else 0.0
                )
                wandb.log({
                    "train/loss": last_loss,
                    "train/loss_ce": float(loss_ce.item()),
                    "train/loss_aux": float(loss_aux.item()),
                    "train/loss_contrast": float(loss_contrast.item()),
                    "train/rc_accept_rate": rc_accept_rate,
                    "train/rc_similarity_mean": rc_similarity_mean,
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/learning_rate": current_lr,  # 실시간 학습률 추적
                }, step=step)
            
            # Validation
            if step % train_config.valid_every == 0:
                model.eval()
                
                with torch.no_grad():
                    preds_all: List[str] = []
                    targets_all: List[str] = []
                    inputs_all: List[str] = []
                    val_loss_total = 0.0
                    val_batch_count = 0
                    
                    for val_batch in val_dataloader:
                        # 원본 저장 (metrics 계산용)
                        original_targets = val_batch["target_text"].copy()
                        original_inputs = val_batch["input_text"].copy()
                        
                        # Reversed sequence training 적용
                        if use_reversed:
                            val_batch_copy = {
                                "input_text": [reverse_arithmetic_numbers(text) for text in val_batch["input_text"]],
                                "target_text": [reverse_arithmetic_numbers(text) for text in val_batch["target_text"]]
                            }
                            val_bt = tokenize_batch(val_batch_copy, input_tokenizer, output_tokenizer)
                        else:
                            val_bt = tokenize_batch(val_batch, input_tokenizer, output_tokenizer)
                        
                        val_src = val_bt.src.to(device)
                        val_tgt_inp = val_bt.tgt_inp.to(device)
                        val_tgt_out = val_bt.tgt_out.to(device)
                        
                        # Validation loss 계산
                        val_logits = model(val_src, val_tgt_inp, input_tokenizer.pad_id)
                        val_loss = loss_fn(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_tgt_out.view(-1),
                        )
                        val_loss_total += val_loss.item()
                        val_batch_count += 1
                        
                        # 생성 (metrics 계산용)
                        gen_ids = model.generate(
                            src=val_src,
                            max_len=train_config.max_gen_len,
                            bos_id=output_tokenizer.bos_id,
                            eos_id=output_tokenizer.eos_id,
                            src_pad_id=input_tokenizer.pad_id,
                        )
                        
                        for i in range(gen_ids.size(0)):
                            seq_chars: List[str] = []
                            for t in gen_ids[i].tolist():
                                idx = int(t)
                                if idx == output_tokenizer.eos_id:
                                    break
                                if idx in output_tokenizer.itos:
                                    ch = output_tokenizer.itos[idx]
                                    if ch.isdigit() or (ch == '-' and not seq_chars):
                                        seq_chars.append(ch)
                            pred_str = "".join(seq_chars)
                            if pred_str == "-":
                                pred_str = ""
                            
                            # Reversed sequence training 결과 복원
                            if use_reversed:
                                pred_str = reverse_arithmetic_numbers(pred_str)
                            
                            preds_all.append(pred_str)
                        
                        # 원본 targets/inputs 사용 (복원된 예측과 비교)
                        targets_all.extend(original_targets)
                        inputs_all.extend(original_inputs)
                    
                    # 평균 validation loss 계산
                    avg_val_loss = val_loss_total / val_batch_count if val_batch_count > 0 else 0.0
                    
                    # 메트릭 계산
                    em_batch = compute_metrics(preds_all, targets_all)
                    current_val_em = float(em_batch.get("EM", -1.0))
                    
                    # Validation 결과 출력
                    print(f"\n{'─'*80}")
                    print(f"[Validation at step {step}] Loss={avg_val_loss:.4f} | EM={em_batch['EM']:.3f} | TES={em_batch['TES']:.3f}")
                    print(f"{'─'*80}")
                    sys.stdout.flush()
                    
                    # wandb에 validation 메트릭 로깅
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "val/EM": em_batch['EM'],
                            "val/TES": em_batch['TES'],
                            "val/EC": em_batch.get('EC', 0.0),
                            "val/RC": em_batch.get('RC', 0.0),
                        }, step=step)
                    
                    # Early stopping 체크 (validation loss 기준)
                    if enable_early_stopping:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0  # 개선되었으므로 카운터 리셋
                        else:
                            patience_counter += 1
                            print(f"⚠️  No improvement for {patience_counter}/{early_stopping_patience} validations (Best Val Loss: {best_val_loss:.4f})")
                            sys.stdout.flush()
                            
                            if patience_counter >= early_stopping_patience:
                                print(f"\n{'='*80}")
                                print(f"🛑 Early stopping triggered at step {step}")
                                print(f"Best validation loss: {best_val_loss:.4f}")
                                print(f"{'='*80}\n")
                                sys.stdout.flush()
                                early_stopped = True
                    
                    # 최고 성능 저장
                    if train_config.save_best_path is not None:
                        if current_val_em > best_em:
                            best_em = current_val_em
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,
                                "model_config": model_config.__dict__,
                                "tokenizer_config": tokenizer_config.__dict__,
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            print(f"🎉 New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")
                            sys.stdout.flush()
                            
                            # wandb에 best model 정보 로깅
                            if use_wandb and WANDB_AVAILABLE:
                                wandb.log({
                                    "best/EM": best_em,
                                    "best/step": step,
                                }, step=step)
                    
                    # 샘플 출력
                    B = len(preds_all)
                    n_show = min(train_config.show_valid_samples, B)
                    print("Sample validation output:")
                    
                    # wandb Table 생성 (샘플 예측 결과)
                    if use_wandb and WANDB_AVAILABLE:
                        table_data = []
                    
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "✓" if pred == tgt else "✗"
                        print(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
                        
                        # wandb table에 추가
                        if use_wandb and WANDB_AVAILABLE:
                            table_data.append([input_str, tgt, pred, "OK" if pred == tgt else "ERR"])
                    
                    # wandb에 예측 샘플 테이블 로깅
                    if use_wandb and WANDB_AVAILABLE and table_data:
                        table = wandb.Table(
                            columns=["Input", "Target", "Prediction", "Status"],
                            data=table_data
                        )
                        wandb.log({"predictions": table}, step=step)
                    
                    print(f"{'─'*80}\n")
                    sys.stdout.flush()
                
                model.train()
                
                # Early stopping이 발동하면 학습 종료
                if early_stopped:
                    break
            
            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                break
            
            # Early stopping이 발동하면 epoch 루프 종료
            if early_stopped:
                break
    
    # 학습 완료
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    
    print(f"\n{'='*80}")
    if early_stopped:
        print(f"🛑 Training stopped early due to no improvement")
    else:
        print(f"✅ Training completed!")
    print(f"Total steps: {step:,} | Total time: {hours}h {mins}m")
    print(f"🏆 Best EM: {best_em:.3f}")
    if enable_early_stopping:
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    sys.stdout.flush()


# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # Command-line arguments 파싱
    parser = argparse.ArgumentParser(description="Transformer 모델 학습")
    parser.add_argument("--user_name", type=str, default="Guest", help="사용자 이름 (wandb run tracking용)")
    parser.add_argument("--use_wandb", action="store_true", help="wandb 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="transformer-arithmetic", help="wandb 프로젝트 이름")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (팀 이름)")
    parser.add_argument("--batch_size", type=int, default=256, help="배치 크기 (기본값: 256, 더 안정적인 학습)")
    parser.add_argument("--lr", type=float, default=2e-4, help="학습률 (기본값: 2e-4, Transformer 권장)")
    parser.add_argument("--num_epochs", type=int, default=30, help="학습 epoch 수")
    parser.add_argument("--d_model", type=int, default=256, help="모델 hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Attention head 수")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Encoder 레이어 수")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Decoder 레이어 수")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience (검증 성능이 개선되지 않는 횟수)")
    parser.add_argument("--no_early_stopping", action="store_true", help="Early stopping 비활성화")
    parser.add_argument("--use_reversed", action="store_true", help="Use reversed sequence training (1의 자리부터 예측)")
    parser.add_argument("--save_name", type=str, default=None, help="모델 저장 이름 (없으면 자동 생성)")
    parser.add_argument("--use_alibi", action="store_true", help="Use ALiBi (Attention with Linear Biases) positional encoding")
    parser.add_argument("--use_nope", action="store_true", help="Use NoPE (No Positional Encoding)")
    # In-stream data augmentation (kang: associative, commutative, boundary, EC, RC)
    parser.add_argument("--prob_associative", type=float, default=0.0, help="Associative augmentation probability")
    parser.add_argument("--prob_commutative", type=float, default=0.0, help="Commutative augmentation probability")
    parser.add_argument("--prob_boundary", type=float, default=0.0, help="Boundary oversample probability")
    parser.add_argument("--prob_ec", type=float, default=0.0, help="Expression consistency (EC) augmentation probability")
    parser.add_argument("--prob_rc", type=float, default=0.0, help="Relational consistency (RC) augmentation probability")
    parser.add_argument("--max_equivalent_variations", type=int, default=5, help="Max equivalent expressions per sample (EC)")
    parser.add_argument("--max_related_variations", type=int, default=4, help="Max related expressions per sample (RC)")
    parser.add_argument("--boundary_max_tries", type=int, default=10, help="Max tries for boundary rejection sampling")
    parser.add_argument("--stage1_epochs", type=int, default=30, help="Stage-1 CE pretrain epochs")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="Stage-2 consistency/contrastive epochs")
    parser.add_argument("--lambda_aux", type=float, default=0.2, help="Weight for auxiliary consistency loss")
    parser.add_argument("--lambda_contrast", type=float, default=0.1, help="Weight for contrastive loss")
    parser.add_argument("--aux_pair_rate", type=float, default=0.5, help="Pair construction rate for assoc/comm aux")
    parser.add_argument("--ec_pair_rate", type=float, default=0.5, help="Pair construction rate for EC positives")
    parser.add_argument("--rc_pair_rate", type=float, default=0.5, help="Pair construction rate for RC negatives")
    parser.add_argument("--boundary_append_rate", type=float, default=0.0, help="Boundary append rate for extra CE")
    parser.add_argument("--boundary_append_samples", type=int, default=1, help="Boundary samples to append per selected example")
    parser.add_argument("--rc_min_similarity", type=float, default=0.72, help="Minimum surface similarity for RC hard negatives")
    parser.add_argument("--rc_max_value_delta", type=int, default=30, help="Maximum absolute target delta for RC hard negatives")
    parser.add_argument("--rc_hard_negative_ratio", type=float, default=1.0, help="Ratio of hard negatives among RC pairs")
    args = parser.parse_args()
    total_epochs = args.stage1_epochs + args.stage2_epochs
    if total_epochs <= 0:
        total_epochs = args.num_epochs
        args.stage1_epochs = total_epochs
        args.stage2_epochs = 0
    
    # ALiBi와 NoPE 동시 사용 방지
    if args.use_alibi and args.use_nope:
        raise ValueError("Cannot use both --use_alibi and --use_nope. Choose one.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # wandb 초기화
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        # Run name 생성
        if args.wandb_run_name:
            # 사용자가 직접 지정한 경우
            run_name = args.wandb_run_name
        else:
            # 자동 생성: Transformer_ep30_d256_lr0.001_bs128_enc4_dec4 형식
            # lr을 소수점 형식으로 변환 (0.001, 0.0005 등)
            if args.lr >= 1:
                lr_str = f"{args.lr:.0f}"
            elif args.lr >= 0.001:
                lr_str = f"{args.lr:.3f}".rstrip('0').rstrip('.')
            elif args.lr >= 0.0001:
                lr_str = f"{args.lr:.4f}".rstrip('0').rstrip('.')
            else:
                lr_str = f"{args.lr:.6f}".rstrip('0').rstrip('.')
            
            run_name = (f"TransformerALiBi_ep{args.num_epochs}_d{args.d_model}_"
                       f"lr{lr_str}_bs{args.batch_size}_"
                       f"enc{args.num_encoder_layers}_dec{args.num_decoder_layers}")
            
            # Reversed sequence training 표시
            if args.use_reversed:
                run_name += "_reversed"
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "model_type": "transformer_alibi",
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_epochs": total_epochs,
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "lambda_aux": args.lambda_aux,
                "lambda_contrast": args.lambda_contrast,
                "device": str(device),
                "positional_encoding": "ALiBi",
                "use_reversed": args.use_reversed,
            }
        )
        print(f"wandb run 시작: {wandb.run.name}")
        print(f"wandb URL: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("경고: wandb를 사용하려 했지만 설치되지 않았습니다. 'pip install wandb' 실행 후 다시 시도하세요.")
        print("wandb 없이 계속 진행합니다...")
    
    # --------------------------------------------------------------------------
    # 1) 데이터 준비
    # --------------------------------------------------------------------------
    train_augment_config = None
    if (
        args.prob_associative > 0 or args.prob_commutative > 0 or args.prob_boundary > 0
        or args.prob_ec > 0 or args.prob_rc > 0
    ):
        train_augment_config = AugmentConfig(
            prob_associative=args.prob_associative,
            prob_commutative=args.prob_commutative,
            prob_boundary=args.prob_boundary,
            prob_ec=args.prob_ec,
            prob_rc=args.prob_rc,
            max_equivalent_variations=args.max_equivalent_variations,
            max_related_variations=args.max_related_variations,
            boundary_max_tries=args.boundary_max_tries,
            aux_pair_rate=args.aux_pair_rate,
            ec_pair_rate=args.ec_pair_rate,
            rc_pair_rate=args.rc_pair_rate,
            boundary_append_rate=args.boundary_append_rate,
            boundary_append_samples=args.boundary_append_samples,
            rc_min_similarity=args.rc_min_similarity,
            rc_max_value_delta=args.rc_max_value_delta,
            rc_hard_negative_ratio=args.rc_hard_negative_ratio,
        )
        print(f"In-stream augmentation: assoc={args.prob_associative}, comm={args.prob_commutative}, "
              f"boundary={args.prob_boundary}, EC={args.prob_ec}, RC={args.prob_rc}")

    train_dataset = ArithmeticDataset(
        num_samples=500_000,
        max_depth=4,
        num_digits=(1, 5),
        seed=123,
        mode="train",
        augment_config=train_augment_config,
    )

    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = ArithmeticDataset(
        num_samples=128,
        max_depth=4,
        num_digits=(1, 5),
        seed=999,
        mode="val",
        augment_config=None,
    )
    
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    
    # --------------------------------------------------------------------------
    # 2) 토크나이저 준비
    # --------------------------------------------------------------------------
    tokenizer_config = TokenizerConfig(
        input_chars=INPUT_CHARS,
        output_chars=OUTPUT_CHARS,
        add_special=True,
    )
    
    input_tokenizer = CharTokenizer(
        tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )
    output_tokenizer = CharTokenizer(
        tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )
    
    # --------------------------------------------------------------------------
    # 3) 모델 설정
    # --------------------------------------------------------------------------
    model_config = ModelConfig(
        model_type="transformer",  # 새로운 모델 타입
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4,  # 일반적으로 d_model의 4배
        dropout=0.1,
    )
    
    # ALiBi 정보 출력
    print("\n" + "="*80)
    print("🔵 ALiBi (Attention with Linear Biases) Positional Encoding")
    print("   - Custom Attention Layer로 완전한 ALiBi 구현")
    print("   - Attention score에 직접 선형 bias 추가")
    print("   - 메모리 효율적이며 긴 시퀀스에 대한 외삽 가능")
    print("   - BLOOM/Hugging Face 구현 참고")
    print("="*80 + "\n")
    
    # --------------------------------------------------------------------------
    # 4) 학습 설정
    # --------------------------------------------------------------------------
    # Best model 저장 경로 생성
    if args.save_name:
        # 사용자가 직접 지정한 경우
        save_best_path = f"best_{args.save_name}.pt"
    else:
        # 자동 생성: 하이퍼파라미터 기반
        if args.lr >= 1:
            lr_str = f"{args.lr:.0f}"
        elif args.lr >= 0.001:
            lr_str = f"{args.lr:.3f}".rstrip('0').rstrip('.')
        elif args.lr >= 0.0001:
            lr_str = f"{args.lr:.4f}".rstrip('0').rstrip('.')
        else:
            lr_str = f"{args.lr:.6f}".rstrip('0').rstrip('.')
        
        model_name = (f"Transformer_ep{total_epochs}_d{args.d_model}_"
                      f"lr{lr_str}_bs{args.batch_size}_"
                      f"enc{args.num_encoder_layers}_dec{args.num_decoder_layers}")
        
        if args.use_reversed:
            model_name += "_reversed"
        
        if args.use_alibi:
            model_name += "_alibi"
        elif args.use_nope:
            model_name += "_nope"
        
        save_best_path = f"best_{model_name}.pt"
    
    print(f"Best model will be saved to: {save_best_path}")
    
    train_config = TrainConfig(
        max_train_steps=None,
        lr=args.lr,
        valid_every=200,
        max_gen_len=48,  # 24 → 48 (긴 숫자 생성 지원)
        show_valid_samples=5,
        num_epochs=total_epochs,
        save_best_path=save_best_path,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lambda_aux=args.lambda_aux,
        lambda_contrast=args.lambda_contrast,
    )
    
    # --------------------------------------------------------------------------
    # 5) 모델 준비
    # --------------------------------------------------------------------------
    print("Creating Encoder-Decoder Transformer model with ALiBi...")
    
    # ALiBi를 사용하는 Transformer 모델
    model = TransformerSeq2SeqWithALiBi(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout,
    )
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # --------------------------------------------------------------------------
    # 6) 학습 시작
    # --------------------------------------------------------------------------
    try:
        train_loop(
            model=model,
            dataloader=train_dataloader,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            device=device,
            val_dataloader=val_dataloader,
            train_config=train_config,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            use_wandb=use_wandb,
            early_stopping_patience=args.early_stopping_patience,
            enable_early_stopping=not args.no_early_stopping,
            use_reversed=args.use_reversed,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            lambda_aux=args.lambda_aux,
            lambda_contrast=args.lambda_contrast,
        )
        
        # --------------------------------------------------------------------------
        # 7) 최종 모델 저장
        # --------------------------------------------------------------------------
        torch.save(model.state_dict(), "transformer_model_final.pt")
        print("Saved transformer_model_final.pt")
        
        # wandb에 최종 모델 아티팩트 저장 (선택적)
        if use_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact("transformer_model", type="model")
            artifact.add_file("transformer_model_final.pt")
            wandb.log_artifact(artifact)
            print("Saved model to wandb artifacts")
    
    finally:
        # wandb 종료
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            print("wandb run 종료")


if __name__ == "__main__":
    main()

