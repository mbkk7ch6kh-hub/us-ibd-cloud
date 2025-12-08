# enrich_smr.py
#
# 역할:
# 1) rs_onil_all_YYYYMMDD.csv (원본 RS 파일, _smr 없는 것만) 중
#    가장 최신 파일을 찾는다.
# 2) 같은 폴더의 smr_factors.csv (SMR 점수/등급, S/M/R raw/pct)와
#    ticker 기준으로 merge한다.
# 3) rs_onil_all_YYYYMMDD_smr.csv 이름으로 저장한다.
#
# 주의:
# - rs_onil_all_YYYYMMDD_smr.csv 같은 "이미 SMR 붙은 파일"은 무시한다.
# - *_smr_smr.csv 같은 괴물 파일은 이 스크립트에서 절대 건드리지 않는다.

import os
import glob
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_base_rs() -> str:
    """
    rs_onil_all_*.csv 중에서 '_smr'가 들어간 파일은 모두 제외하고,
    (즉, SMR이 아직 안 붙은 순수 RS 파일만 남긴 뒤)
    그 중 가장 최신 파일(이름 기준)을 반환한다.
    """
    pattern = os.path.join(BASE_DIR, "rs_onil_all_*.csv")
    files = glob.glob(pattern)

    # '_smr'가 이름에 포함된 것은 이미 SMR 붙인 결과물이므로 제외
    base_files = [f for f in files if "_smr" not in os.path.basename(f)]

    if not base_files:
        raise FileNotFoundError("rs_onil_all_*.csv (원본 RS 파일)을 찾지 못했습니다.")

    base_files.sort()
    latest = base_files[-1]
    print(f"[INFO] 원본 RS 파일 선택: {os.path.basename(latest)}")
    return latest


def load_smr_factors() -> pd.DataFrame:
    """
    같은 폴더에 있는 smr_factors.csv 파일을 읽어온다.
    이 파일은 이전 단계에서 만들어 둔 SMR 지표 파일이라고 가정.
    """
    smr_path = os.path.join(BASE_DIR, "smr_factors.csv")
    if not os.path.exists(smr_path):
        raise FileNotFoundError(f"SMR 파일을 찾을 수 없습니다: {smr_path}")

    df = pd.read_csv(smr_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"ticker", "s_raw", "m_raw", "r_raw", "s_pct", "m_pct", "r_pct", "smr_score", "smr_grade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"smr_factors.csv에 필요한 컬럼이 없습니다: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    print(f"[INFO] SMR 파일 로드: {os.path.basename(smr_path)}, 종목 수: {len(df)}개")
    return df


def merge_rs_and_smr(rs_path: str, smr_df: pd.DataFrame) -> pd.DataFrame:
    """
    RS 결과 파일(rs_onil_all_YYYYMMDD.csv)과 SMR 팩터를 merge하여
    최종 df를 반환한다.
    """
    rs_df = pd.read_csv(rs_path)
    rs_df.columns = [c.strip().lower() for c in rs_df.columns]

    if "ticker" not in rs_df.columns:
        raise ValueError("RS 파일에 'ticker' 컬럼이 없습니다.")

    rs_df["ticker"] = rs_df["ticker"].astype(str).str.upper()

    print(f"[INFO] RS 파일 로드: {os.path.basename(rs_path)}, 종목 수: {len(rs_df)}개")

    merged = pd.merge(
        rs_df,
        smr_df,
        on="ticker",
        how="left",
        suffixes=("", "_smrdup"),
    )

    # 혹시라도 중복 접미사 붙은 컬럼 있으면 정리
    dup_cols = [c for c in merged.columns if c.endswith("_smrdup")]
    if dup_cols:
        print(f"[INFO] 중복 SMR 컬럼 제거: {dup_cols}")
        merged = merged.drop(columns=dup_cols)

    print(f"[INFO] merge 완료. 최종 종목 수: {len(merged)}개")
    return merged


def build_output_path(rs_path: str) -> str:
    """
    입력 RS 파일명(rs_onil_all_YYYYMMDD.csv)에서 날짜 부분을 추출해
    rs_onil_all_YYYYMMDD_smr.csv 형태의 출력 경로를 만든다.
    """
    base = os.path.basename(rs_path)  # 예: rs_onil_all_20251207.csv
    name, ext = os.path.splitext(base)

    # "rs_onil_all_" 이후 8자리(YYYYMMDD)를 잡는다.
    # 혹시 형식이 조금 달라도 안전하게 처리하려고 함.
    date_part = ""
    prefix = "rs_onil_all_"
    if name.startswith(prefix):
        rest = name[len(prefix) :]  # 20251207
        date_part = rest[:8]
    else:
        # 형식이 다를 경우 그냥 전체 이름에 _smr 붙인다.
        date_part = ""

    if date_part and len(date_part) == 8:
        out_name = f"rs_onil_all_{date_part}_smr.csv"
    else:
        out_name = f"{name}_smr.csv"

    out_path = os.path.join(BASE_DIR, out_name)
    print(f"[INFO] 출력 파일명: {out_name}")
    return out_path


def main():
    print("=== RS + SMR 병합 시작 (enrich_smr.py) ===")

    # 1) 원본 RS 파일 찾기 (_smr 없는 것만)
    rs_path = find_latest_base_rs()

    # 2) SMR 팩터 로딩
    smr_df = load_smr_factors()

    # 3) merge
    merged_df = merge_rs_and_smr(rs_path, smr_df)

    # 4) 출력 경로 결정 & 저장
    out_path = build_output_path(rs_path)
    merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] 최종 파일 저장: {out_path}")
    print("=== RS + SMR 병합 완료 ===")


if __name__ == "__main__":
    main()
