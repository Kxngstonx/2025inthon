"""베이스라인 데이터 생성 및 DataLoader"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Union, Optional
import random
from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation
from augmentation import AugmentConfig, augment_sample, build_training_bundle


def _rand_int(rng: random.Random, num_digits: Tuple[int, int]) -> int:
    """랜덤 정수 생성 (선행 0 없음)"""
    lo, hi = num_digits
    n = rng.randint(lo, hi)
    if n == 1:
        return rng.randint(0, 9)
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(n - 1)]
    return int(str(first) + "".join(str(x) for x in rest))


def _gen_expr(rng: random.Random, depth: int, num_digits: Tuple[int, int]) -> Tuple[str, int]:
    """재귀적 수식 생성 (연산자 우선순위 반영)"""
    
    def number() -> Tuple[str, int]:
        v = _rand_int(rng, num_digits)
        return str(v), v
    
    def factor(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.7:
            return number()
        e, v = expr(d - 1)
        return f"({e})", v
    
    def term(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.5:
            return factor(d)
        
        op = rng.choice(["*", "//"])
        le, lv = term(rng.randint(0, d - 1))
        re, rv = factor(rng.randint(0, d - 1))
        
        if op == "//" and rv == 0:
            rv = rng.randint(1, 9)
            re = str(rv)
        
        v = lv // rv if op == "//" else lv * rv
        return f"{le}{op}{re}", v
    
    def expr(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.5:
            return term(d)
        
        op = rng.choice(["+", "-"])
        le, lv = expr(rng.randint(0, d - 1))
        re, rv = term(rng.randint(0, d - 1))
        
        # 음수 방지
        if op == "-" and lv < rv:
            lv, rv = rv, lv
            le, re = re, le
        
        v = lv + rv if op == "+" else lv - rv
        return f"{le}{op}{re}", v
    
    return expr(depth)


class ArithmeticDataset(Dataset):
    """사칙연산 수식 생성 Dataset (optional in-stream augmentation)."""

    def __init__(
        self,
        num_samples: int,
        max_depth: int = 2,
        num_digits: Tuple[int, int] = (1, 3),
        seed: int = 42,
        mode: str = "train",
        augment_config: Optional["AugmentConfig"] = None,
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_digits = num_digits
        self.seed = seed
        self.mode = mode
        self.augment_config = augment_config

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        depth = rng.randint(0, self.max_depth)
        base_expr, val = _gen_expr(rng, depth, self.num_digits)
        base_target = str(val)
        expr, target = base_expr, base_target

        if self.augment_config is not None and self.augment_config.is_enabled():
            expr, target = augment_sample(
                base_expr,
                base_target,
                rng,
                self.augment_config,
                _gen_expr,
                self.num_digits,
                self.max_depth,
            )

        sample: Dict[str, Any] = {"input_text": expr, "target_text": target, "meta": {"depth": depth}}
        if self.mode == "train" and self.augment_config is not None and self.augment_config.is_enabled():
            bundle = build_training_bundle(
                expr=base_expr,
                target=base_target,
                rng=rng,
                config=self.augment_config,
                gen_expr_fn=_gen_expr,
                num_digits=self.num_digits,
                max_depth=self.max_depth,
            )
            sample["aux_pairs"] = bundle["aux_pairs"]
            sample["ec_pairs"] = bundle["ec_pairs"]
            sample["rc_pairs"] = bundle["rc_pairs"]
            sample["boundary_extra"] = bundle["boundary_extra"]
            sample["rc_quality"] = bundle["rc_quality"]
        return sample


def _pair_aware_train_collate(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    batch = collate_fn_with_validation(samples, is_training=True)
    batch["aux_pairs"] = [s.get("aux_pairs", []) for s in samples]
    batch["ec_pairs"] = [s.get("ec_pairs", []) for s in samples]
    batch["rc_pairs"] = [s.get("rc_pairs", []) for s in samples]
    batch["boundary_extra"] = [s.get("boundary_extra", []) for s in samples]
    batch["rc_quality"] = [s.get("rc_quality", {}) for s in samples]
    return batch


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """DataLoader 생성 (mode="train"일 때 자동 검증)"""
    is_training = (getattr(dataset, "mode", mode) == "train")
    collate_fn = _pair_aware_train_collate if is_training else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
