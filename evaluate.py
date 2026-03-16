"""
best_model.pt 모델 평가 스크립트

CSV 파일을 입력받아 모델 성능을 평가합니다.
- Depth별 성능
- 출력 길이(len)별 성능
- 괄호 유무별 성능

사용법:
    python evaluate.py input_data.csv
    python evaluate.py input_data.csv --output results.csv
    python evaluate.py input_data.csv --num-samples 5000
"""

import torch
from model import Model
import pandas as pd
from tqdm import tqdm
import sys
import os
import argparse
from typing import Optional, Dict, List
from do_not_edit.metric import compute_metrics, exact_match


# ======================================================================================
# 데이터 로드
# ======================================================================================

def load_data_from_csv(csv_path: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    CSV 파일에서 데이터를 로드
    
    Args:
        csv_path: CSV 파일 경로
        num_samples: 로드할 샘플 수 (None이면 전체)
    
    Returns:
        DataFrame
    """
    print(f"\n[데이터 로드]")
    print(f"  파일 경로: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  ✅ {len(df):,}개 행 로드 완료")
    print(f"  컬럼: {list(df.columns)}")
    
    # 필수 컬럼 확인
    required_cols = ["input_text", "target_text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    # num_samples가 지정되면 샘플링
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"  📊 {num_samples:,}개 샘플로 제한")
    
    return df


# ======================================================================================
# 모델 로드 및 예측
# ======================================================================================

def load_model(model_path: str = "best_model.pt") -> Model:
    """
    모델 로드
    
    Args:
        model_path: 모델 파일 경로
    
    Returns:
        Model 인스턴스
    """
    print(f"\n[모델 로드]")
    print(f"  모델 경로: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # model_path를 Model 생성자에 전달
    model = Model(model_path=model_path)
    print(f"  ✅ 모델 로드 성공!")
    print(f"  디바이스: {model.device}")
    print(f"  최대 생성 길이: {model.max_len}")
    
    return model


def predict_batch(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """
    배치 예측 수행
    
    Args:
        model: 모델 인스턴스
        df: 입력 데이터프레임
    
    Returns:
        예측 결과가 추가된 데이터프레임
    """
    print(f"\n[예측 수행]")
    print(f"  총 {len(df):,}개 샘플 예측 시작...")
    
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  진행", ncols=80, leave=False, mininterval=0.5, position=0):
        input_text = str(row["input_text"])
        try:
            pred = model.predict(input_text)
            predictions.append(pred)
        except Exception as e:
            predictions.append("")
    
    df["prediction"] = predictions
    print(f"  ✅ 예측 완료! ({len(df):,}개)")
    
    return df


# ======================================================================================
# 특징 추출
# ======================================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터에서 분석용 특징 추출
    
    Args:
        df: 데이터프레임 (input_text, target_text, prediction 컬럼 필요)
    
    Returns:
        특징이 추가된 데이터프레임
    """
    print(f"\n[특징 추출]")
    
    # 문자열 변환 및 NaN 처리
    for col in ["input_text", "target_text", "prediction"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(["nan", "None", "NaN"], "")
    
    # depth (이미 있으면 사용, 없으면 0)
    if "depth" not in df.columns:
        df["depth"] = 0
        print(f"  ⚠️ 'depth' 컬럼이 없습니다. 모두 0으로 설정합니다.")
    
    # 출력 길이
    df["target_len"] = df["target_text"].str.len()
    
    # 괄호 유무
    df["has_paren"] = df["input_text"].str.contains(r"\(", regex=True).astype(int)
    
    # 정답 여부
    df["is_correct"] = df.apply(
        lambda row: exact_match(row["prediction"], row["target_text"]), 
        axis=1
    )
    
    print(f"  ✅ 특징 추출 완료")
    print(f"    - depth: {df['depth'].nunique()}개 값")
    print(f"    - target_len: {df['target_len'].min()}~{df['target_len'].max()}")
    print(f"    - has_paren: {df['has_paren'].sum():,}개 ({df['has_paren'].mean()*100:.1f}%)")
    
    return df


# ======================================================================================
# 성능 분석
# ======================================================================================

def analyze_overall_performance(df: pd.DataFrame, consistency_type: Optional[str] = None):
    """전체 성능 분석"""
    print(f"\n{'='*80}")
    print(f"전체 성능")
    print(f"{'='*80}")
    
    preds = df["prediction"].tolist()
    targets = df["target_text"].tolist()
    
    # group_id가 있으면 전달
    group_ids = None
    if "group_id" in df.columns:
        group_ids = df["group_id"].tolist()
    
    # BOTH 모드: EC와 RC를 따로 계산
    if consistency_type == "BOTH" and group_ids is not None:
        print(f"  샘플 수: {len(df):,}")
        
        # 기본 metric (EM, TES)
        metrics = compute_metrics(preds, targets, group_ids)
        print(f"  EM (Exact Match): {metrics['EM']*100:.2f}%")
        print(f"  TES (Token Edit Similarity): {metrics['TES']*100:.2f}%")
        
        # EC 계산: 각 그룹에서 2개씩만 사용
        print(f"\n  [EC 계산 (각 그룹 2개씩)]")
        ec_indices = []
        for group_id in set(group_ids):
            group_indices = [i for i, g in enumerate(group_ids) if g == group_id]
            if len(group_indices) >= 2:
                ec_indices.extend(group_indices[:2])
        
        ec_preds = [preds[i] for i in ec_indices]
        ec_targets = [targets[i] for i in ec_indices]
        ec_group_ids = [group_ids[i] for i in ec_indices]
        ec_metrics = compute_metrics(ec_preds, ec_targets, ec_group_ids)
        print(f"    EC (Equational Consistency): {ec_metrics['EC']*100:.2f}%")
        print(f"    (사용 샘플: {len(ec_indices):,}개)")
        
        # RC 계산: 각 그룹에서 3개 이상만 사용
        print(f"\n  [RC 계산 (각 그룹 3개 이상)]")
        rc_indices = []
        for group_id in set(group_ids):
            group_indices = [i for i, g in enumerate(group_ids) if g == group_id]
            if len(group_indices) >= 3:
                rc_indices.extend(group_indices)
        
        rc_preds = [preds[i] for i in rc_indices]
        rc_targets = [targets[i] for i in rc_indices]
        rc_group_ids = [group_ids[i] for i in rc_indices]
        rc_metrics = compute_metrics(rc_preds, rc_targets, rc_group_ids)
        print(f"    RC (Reasoning Consistency): {rc_metrics['RC']*100:.2f}%")
        print(f"    (사용 샘플: {len(rc_indices):,}개)")
        
        # consistency 통계
        unique_groups = len(set(g for g in group_ids if g is not None))
        print(f"\n  [Consistency 통계]")
        print(f"    고유 그룹 수: {unique_groups:,}")
        print(f"    평균 반복 횟수: {len(df) / unique_groups if unique_groups > 0 else 0:.1f}회")
    else:
        # 일반 모드
        metrics = compute_metrics(preds, targets, group_ids)
        
        print(f"  샘플 수: {len(df):,}")
        print(f"  EM (Exact Match): {metrics['EM']*100:.2f}%")
        print(f"  TES (Token Edit Similarity): {metrics['TES']*100:.2f}%")
        
        # EC, RC는 group_ids가 있을 때만 의미 있음
        if group_ids is not None:
            print(f"  EC (Equational Consistency): {metrics['EC']*100:.2f}%")
            print(f"  RC (Reasoning Consistency): {metrics['RC']*100:.2f}%")
            
            # consistency 통계
            unique_groups = len(set(g for g in group_ids if g is not None))
            print(f"\n  [Consistency 통계]")
            print(f"    고유 그룹 수: {unique_groups:,}")
            print(f"    평균 반복 횟수: {len(df) / unique_groups if unique_groups > 0 else 0:.1f}회")


def analyze_by_depth(df: pd.DataFrame):
    """Depth별 성능 분석"""
    print(f"\n{'='*80}")
    print(f"Depth별 성능")
    print(f"{'='*80}")
    
    has_group_id = "group_id" in df.columns
    
    depth_results = []
    for depth in sorted(df["depth"].unique()):
        depth_df = df[df["depth"] == depth]
        preds = depth_df["prediction"].tolist()
        targets = depth_df["target_text"].tolist()
        group_ids = depth_df["group_id"].tolist() if has_group_id else None
        metrics = compute_metrics(preds, targets, group_ids)
        
        depth_results.append({
            "depth": depth,
            "count": len(depth_df),
            "EM": metrics["EM"],
            "TES": metrics["TES"],
            "EC": metrics.get("EC", 0.0),
            "RC": metrics.get("RC", 0.0),
        })
    
    if has_group_id:
        print(f"{'Depth':<10} {'Count':<10} {'EM':<12} {'TES':<12} {'EC':<12} {'RC':<12}")
        print("-" * 70)
        for result in depth_results:
            print(f"{int(result['depth']):<10} {result['count']:<10,} "
                  f"{result['EM']*100:<12.2f}% {result['TES']*100:<12.2f}% "
                  f"{result['EC']*100:<12.2f}% {result['RC']*100:<12.2f}%")
    else:
        print(f"{'Depth':<10} {'Count':<15} {'EM':<15} {'TES':<15}")
        print("-" * 60)
        for result in depth_results:
            print(f"{int(result['depth']):<10} {result['count']:<15,} "
                  f"{result['EM']*100:<15.2f}% {result['TES']*100:<15.2f}%")


def analyze_by_target_len(df: pd.DataFrame):
    """출력 길이별 성능 분석"""
    print(f"\n{'='*80}")
    print(f"출력 길이(Target Length)별 성능")
    print(f"{'='*80}")
    
    has_group_id = "group_id" in df.columns
    
    len_results = []
    for target_len in sorted(df["target_len"].unique()):
        len_df = df[df["target_len"] == target_len]
        preds = len_df["prediction"].tolist()
        targets = len_df["target_text"].tolist()
        group_ids = len_df["group_id"].tolist() if has_group_id else None
        metrics = compute_metrics(preds, targets, group_ids)
        
        len_results.append({
            "target_len": target_len,
            "count": len(len_df),
            "EM": metrics["EM"],
            "TES": metrics["TES"],
            "EC": metrics.get("EC", 0.0),
            "RC": metrics.get("RC", 0.0),
        })
    
    if has_group_id:
        print(f"{'Length':<10} {'Count':<10} {'EM':<12} {'TES':<12} {'EC':<12} {'RC':<12}")
        print("-" * 70)
        for result in len_results:
            print(f"{int(result['target_len']):<10} {result['count']:<10,} "
                  f"{result['EM']*100:<12.2f}% {result['TES']*100:<12.2f}% "
                  f"{result['EC']*100:<12.2f}% {result['RC']*100:<12.2f}%")
    else:
        print(f"{'Length':<10} {'Count':<15} {'EM':<15} {'TES':<15}")
        print("-" * 60)
        for result in len_results:
            print(f"{int(result['target_len']):<10} {result['count']:<15,} "
                  f"{result['EM']*100:<15.2f}% {result['TES']*100:<15.2f}%")


def analyze_by_parentheses(df: pd.DataFrame):
    """괄호 유무별 성능 분석"""
    print(f"\n{'='*80}")
    print(f"괄호 유무별 성능")
    print(f"{'='*80}")
    
    has_group_id = "group_id" in df.columns
    
    paren_results = []
    for has_paren in [0, 1]:
        paren_df = df[df["has_paren"] == has_paren]
        if len(paren_df) == 0:
            continue
        
        preds = paren_df["prediction"].tolist()
        targets = paren_df["target_text"].tolist()
        group_ids = paren_df["group_id"].tolist() if has_group_id else None
        metrics = compute_metrics(preds, targets, group_ids)
        
        label = "괄호 있음" if has_paren else "괄호 없음"
        paren_results.append({
            "label": label,
            "count": len(paren_df),
            "EM": metrics["EM"],
            "TES": metrics["TES"],
            "EC": metrics.get("EC", 0.0),
            "RC": metrics.get("RC", 0.0),
        })
    
    if has_group_id:
        print(f"{'괄호':<15} {'Count':<10} {'EM':<12} {'TES':<12} {'EC':<12} {'RC':<12}")
        print("-" * 75)
        for result in paren_results:
            print(f"{result['label']:<15} {result['count']:<10,} "
                  f"{result['EM']*100:<12.2f}% {result['TES']*100:<12.2f}% "
                  f"{result['EC']*100:<12.2f}% {result['RC']*100:<12.2f}%")
    else:
        print(f"{'괄호':<15} {'Count':<15} {'EM':<15} {'TES':<15}")
        print("-" * 60)
        for result in paren_results:
            print(f"{result['label']:<15} {result['count']:<15,} "
                  f"{result['EM']*100:<15.2f}% {result['TES']*100:<15.2f}%")


def show_error_examples(df: pd.DataFrame, n: int = 10):
    """오답 예시 출력"""
    wrong_df = df[df["is_correct"] == 0]
    
    if len(wrong_df) == 0:
        print(f"\n✅ 모든 예측이 정확합니다!")
        return
    
    print(f"\n{'='*80}")
    print(f"오답 예시 (최대 {n}개)")
    print(f"{'='*80}")
    
    for _, row in wrong_df.head(n).iterrows():
        depth = row.get("depth", "?")
        has_paren = "괄호O" if row.get("has_paren", 0) else "괄호X"
        print(f"  [Depth {depth}, {has_paren}] {row['input_text']} = {row['target_text']} "
              f"(예측: {row['prediction']})")


# ======================================================================================
# 메인 함수
# ======================================================================================

def evaluate(
    input_csv: str,
    model_path: str = "best_model.pt",
    output_csv: Optional[str] = None,
    num_samples: Optional[int] = None,
    show_errors: bool = True,
    consistency_type: Optional[str] = None,
):
    """
    모델 평가 메인 함수
    
    Args:
        input_csv: 입력 CSV 파일 경로
        model_path: 모델 파일 경로
        output_csv: 결과 저장 CSV 경로 (None이면 저장 안 함)
        num_samples: 평가할 샘플 수 (None이면 전체)
        show_errors: 오답 예시 출력 여부
        consistency_type: Consistency 테스트 타입
            - None: 전체 데이터 사용 (기본값)
            - "EC": 각 그룹에서 2개씩만 선택 (EC 측정)
            - "RC": 3개 이상인 그룹만 유지 (RC 측정)
            - "BOTH": 전체 데이터 사용, EC/RC 따로 계산
    """
    print("="*80)
    print("모델 평가 시작")
    print("="*80)
    
    # 1. 데이터 로드
    df = load_data_from_csv(input_csv, num_samples)
    
    # 1.5. Consistency 타입에 따라 데이터 필터링
    if consistency_type is not None:
        if "group_id" not in df.columns:
            print(f"⚠️ 'group_id' 컬럼이 없습니다. consistency_type을 무시합니다.")
        else:
            consistency_type_upper = consistency_type.upper()
            if consistency_type_upper == "BOTH":
                # BOTH: 전체 데이터 사용 (나중에 분석 시 분리)
                print(f"\n[Consistency 모드: BOTH]")
                print(f"  전체 데이터 사용 (분석 시 EC/RC 따로 계산)")
                print(f"  샘플 수: {len(df):,}")
            elif consistency_type_upper == "EC":
                # EC: 각 그룹에서 2개만 선택
                print(f"\n[Consistency 필터링: EC]")
                print(f"  각 그룹에서 2개씩만 선택")
                
                selected_rows = []
                for group_id in df["group_id"].unique():
                    group_df = df[df["group_id"] == group_id]
                    # 각 그룹에서 최대 2개 선택
                    selected = group_df.head(2)
                    selected_rows.append(selected)
                
                df = pd.concat(selected_rows, ignore_index=True)
                print(f"  필터링 후 샘플 수: {len(df):,}")
                
            elif consistency_type_upper == "RC":
                # RC: 각 그룹에서 3개 이상만 유지
                print(f"\n[Consistency 필터링: RC]")
                print(f"  각 그룹에서 3개 이상만 유지 (3개 미만 그룹 제거)")
                
                selected_rows = []
                for group_id in df["group_id"].unique():
                    group_df = df[df["group_id"] == group_id]
                    # 3개 이상인 그룹만 유지 (전부 사용)
                    if len(group_df) >= 3:
                        selected_rows.append(group_df)
                
                if selected_rows:
                    df = pd.concat(selected_rows, ignore_index=True)
                else:
                    print(f"  ⚠️ 3개 이상인 그룹이 없습니다.")
                    df = pd.DataFrame()  # 빈 데이터프레임
                
                print(f"  필터링 후 샘플 수: {len(df):,}")
                
            else:
                print(f"⚠️ 알 수 없는 consistency_type: {consistency_type}. 무시합니다.")
    
    # 2. 모델 로드
    model = load_model(model_path)
    
    # 3. 예측
    df = predict_batch(model, df)
    
    # 4. 특징 추출
    df = extract_features(df)
    
    # 5. 성능 분석
    analyze_overall_performance(df, consistency_type)
    analyze_by_depth(df)
    analyze_by_target_len(df)
    analyze_by_parentheses(df)
    
    # 6. 오답 예시
    if show_errors:
        show_error_examples(df, n=10)
    
    # 7. 결과 저장
    if output_csv:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ 결과 저장: {output_csv}")
    
    print(f"\n{'='*80}")
    print(f"평가 완료!")
    print(f"{'='*80}")
    
    return df


def main():
    """명령줄 인터페이스"""
    parser = argparse.ArgumentParser(description="모델 평가 스크립트")
    parser.add_argument(
        "input_csv",
        type=str,
        help="입력 CSV 파일 경로 (input_text, target_text 컬럼 필요)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pt",
        help="모델 파일 경로 (기본값: best_model.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 CSV 파일 경로 (지정하지 않으면 저장 안 함)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="평가할 샘플 수 (지정하지 않으면 전체)"
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="오답 예시를 출력하지 않음"
    )
    parser.add_argument(
        "--consistency-type",
        type=str,
        choices=["EC", "RC", "BOTH", "ec", "rc", "both"],
        default=None,
        help="Consistency 테스트 타입: EC (2-way), RC (3-way), BOTH (둘 다)"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate(
            input_csv=args.input_csv,
            model_path=args.model,
            output_csv=args.output,
            num_samples=args.num_samples,
            show_errors=not args.no_errors,
            consistency_type=args.consistency_type,
        )
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ======================================================================================
# 사용 예시
# ======================================================================================

if __name__ == "__main__":
    # 명령줄에서 실행
    # python evaluate.py input_data.csv
    # python evaluate.py input_data.csv --output results.csv
    # python evaluate.py input_data.csv --num-samples 5000
    # python evaluate.py generated_data/val_high_paren_all.csv --output eval_high_paren.csv
    
    # Consistency 테스트 (group_id가 포함된 CSV 사용)
    # 1단계: 충분히 많이 반복해서 생성 (예: 5회)
    # python generate.py --name val_baseline --consistency-repeat 5
    
    # 2단계: 평가 시 타입 선택
    # python evaluate.py generated_data/val_baseline_all.csv --consistency-type EC
    #   → 각 그룹에서 2개씩만 선택 → EC만 측정
    # python evaluate.py generated_data/val_baseline_all.csv --consistency-type RC
    #   → 3개 이상인 그룹만 유지 (전부 사용) → RC만 측정
    # python evaluate.py generated_data/val_baseline_all.csv --consistency-type BOTH
    #   → 전체 사용 → EC, RC 둘 다 측정 (각각 다르게 계산)
    # python evaluate.py generated_data/val_baseline_all.csv
    #   → 전체 사용 → 그룹 크기에 따라 EC/RC 측정됨
    
    main()

