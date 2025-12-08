# calc_rs_onil.py
#
# 기능 요약
# 1) prices/{ticker}.csv 에서 각 종목의 3/6/9/12개월 수익률 계산
# 2) 오닐식 가중 12M 수익률로 개별 RS (0~100, 1~99) 계산
# 3) us_universe.csv 의 sector / industry 기반 group_key 생성
# 4) group_key 기준 6M 평균 수익률로 산업군 RS/랭크/등급 계산
# 5) 최근 50일 평균 거래량(avg_vol_50), 평균 거래대금(avg_dollar_vol_50) 계산
# 6) 결과 파일:
#    - rs_onil_all_YYYYMMDD.csv (종목 레벨 전체)
#    - industry_rs_6m_YYYYMMDD.csv (산업군 레벨)
#    - rs_onil_watchlist_YYYYMMDD.csv (RS 상위 300개 예시)

import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd

PRICES_DIR = "prices"
UNIVERSE_FILE = "us_universe.csv"


# ---------- 유틸 ----------

def load_universe() -> pd.DataFrame:
    """us_universe.csv 로드, ticker/sector/industry 메타데이터 리턴"""
    if not os.path.exists(UNIVERSE_FILE):
        print(f"[WARN] {UNIVERSE_FILE} 파일이 없습니다. 산업군 정보 없이 진행합니다.")
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    # ticker 컬럼 찾기
    if "ticker" in df.columns:
        ticker_col = "ticker"
    elif "symbol" in df.columns:
        ticker_col = "symbol"
    else:
        print(f"[WARN] us_universe.csv 에 ticker/symbol 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = df.rename(columns={ticker_col: "ticker"})

    keep_cols = ["ticker"]
    if "sector" in df.columns:
        keep_cols.append("sector")
    if "industry" in df.columns:
        keep_cols.append("industry")

    meta = df[keep_cols].copy()
    meta["ticker"] = meta["ticker"].astype(str).str.upper()

    return meta


def load_price_df(ticker: str) -> pd.DataFrame | None:
    """prices/{ticker}.csv 로드해서 date 기준 정렬"""
    path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_onil_metrics(ticker: str, df: pd.DataFrame) -> dict | None:
    """
    오닐식 12M 가중 수익률 계산
    - 최근 3M, 6M, 9M, 12M 수익률
    - 가중치: 40%, 20%, 20%, 20%
    - 6M 수익률(ret_6m)은 산업군 RS용으로도 사용
    - 최근 50일 평균 거래량, 평균 거래대금까지 계산
    """

    # 최소 12M(대략 252거래일) 없으면 스킵
    if len(df) < 252:
        return None

    last_row = df.iloc[-1]
    last_date = last_row["date"]
    last_close = float(last_row["close"])

    def price_n_days_ago(days: int) -> float | float:
        idx = len(df) - 1 - days
        if idx < 0:
            return np.nan
        return float(df.iloc[idx]["close"])

    # 대략 3M(63), 6M(126), 9M(189), 12M(252) 거래일 기준
    p_3m = price_n_days_ago(63)
    p_6m = price_n_days_ago(126)
    p_9m = price_n_days_ago(189)
    p_12m = price_n_days_ago(252)

    def safe_ret(base_price: float) -> float:
        if base_price is None or np.isnan(base_price) or base_price <= 0:
            return np.nan
        return last_close / base_price - 1.0

    ret_3m = safe_ret(p_3m)
    ret_6m = safe_ret(p_6m)
    ret_9m = safe_ret(p_9m)
    ret_12m = safe_ret(p_12m)

    # 4개 수익률 중 하나라도 NaN이면 스킵
    if any(np.isnan(x) for x in [ret_3m, ret_6m, ret_9m, ret_12m]):
        return None

    # 오닐식 가중 12M 수익률 (3M 40%, 나머지 20%)
    onil_weighted = 0.4 * ret_3m + 0.2 * (ret_6m + ret_9m + ret_12m)

    # 최근 50일 평균 거래량
    avg_vol_50 = np.nan
    if "volume" in df.columns:
        avg_vol_50 = float(df["volume"].tail(50).mean())

    # 최근 50일 평균 거래대금 (달러)
    avg_dollar_vol_50 = np.nan
    if not np.isnan(avg_vol_50):
        avg_dollar_vol_50 = last_close * avg_vol_50  # USD/day

    return {
        "ticker": ticker,
        "last_date": last_date,
        "last_close": last_close,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_9m": ret_9m,
        "ret_12m": ret_12m,
        "onil_weighted_ret": onil_weighted,
        "avg_vol_50": avg_vol_50,
        "avg_dollar_vol_50": avg_dollar_vol_50,
    }


def attach_industry_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    us_universe.csv 에서 sector/industry 정보를 붙이고 group_key 생성
    group_key = industry 우선, 없으면 sector, 둘 다 없으면 'Unknown'
    """
    meta = load_universe()

    if meta.empty:
        print("[WARN] 산업군 메타데이터가 없어 group_key를 'Unknown'으로 처리합니다.")
        df["sector"] = None
        df["industry"] = None
        df["group_key"] = "Unknown"
        return df

    merged = df.merge(meta, on="ticker", how="left")

    has_sector = "sector" in merged.columns
    has_industry = "industry" in merged.columns

    if has_industry and has_sector:
        group_key = merged["industry"].fillna(merged["sector"]).fillna("Unknown")
    elif has_industry:
        group_key = merged["industry"].fillna("Unknown")
    elif has_sector:
        group_key = merged["sector"].fillna("Unknown")
    else:
        group_key = pd.Series(["Unknown"] * len(merged))

    merged["group_key"] = group_key
    return merged


def calc_group_rs(df: pd.DataFrame) -> pd.DataFrame:
    """
    종목 레벨(df)에 포함된 ret_6m, group_key 를 이용해
    오닐식 산업군 랭크/RS/등급 계산.

    반환 컬럼:
      - group_key
      - n_members
      - avg_ret_6m
      - group_rank  : 1 = 가장 강한 산업군
      - group_rs_99 : 1~99 점수 (상위일수록 높음)
      - group_rs_100: 0~100 (퍼센타일)
      - group_grade : A~E (상위 20% 단위)
    """
    valid = df.dropna(subset=["group_key", "ret_6m"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "group_key",
                "n_members",
                "avg_ret_6m",
                "group_rank",
                "group_rs_99",
                "group_rs_100",
                "group_grade",
            ]
        )

    group = (
        valid.groupby("group_key")
        .agg(
            n_members=("ticker", "nunique"),
            avg_ret_6m=("ret_6m", "mean"),
        )
        .reset_index()
    )

    # 성과 기준 내림차순 -> 1위가 가장 강한 산업군
    group = group.sort_values("avg_ret_6m", ascending=False).reset_index(drop=True)

    # 산업군 순위: 1 = 가장 강함
    group["group_rank"] = group["avg_ret_6m"].rank(
        method="dense", ascending=False
    ).astype(int)

    # RS 스케일: 상위 산업군일수록 큰 값
    # percentile rank (ascending=True → 수익률 높을수록 pct ↑)
    pct = group["avg_ret_6m"].rank(pct=True, ascending=True)
    group["group_rs_99"] = (pct * 98 + 1).round().astype(int)
    group["group_rs_100"] = (pct * 100).round(1)

    # A~E 등급 (상위 20% = A)
    n_groups = len(group)

    def grade_from_rank(rank: int) -> str:
        ratio = rank / n_groups
        if ratio <= 0.2:
            return "A"
        elif ratio <= 0.4:
            return "B"
        elif ratio <= 0.6:
            return "C"
        elif ratio <= 0.8:
            return "D"
        else:
            return "E"

    group["group_grade"] = group["group_rank"].apply(grade_from_rank)

    return group


# ---------- 메인 ----------

def main():
    # 1) universe / ticker 목록 준비
    meta = load_universe()

    if not meta.empty:
        tickers = sorted(meta["ticker"].dropna().unique().tolist())
    else:
        price_files = glob(os.path.join(PRICES_DIR, "*.csv"))
        tickers = sorted(
            [os.path.splitext(os.path.basename(p))[0].upper() for p in price_files]
        )

    print(f"[INFO] RS 계산 대상 티커 수: {len(tickers)}")

    rows = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"({i}/{len(tickers)}) {ticker} RS 계산 중...")
        df_price = load_price_df(ticker)
        if df_price is None:
            print(f"  [SKIP] {ticker}: 가격 데이터 없음 또는 형식 오류")
            continue

        try:
            metrics = compute_onil_metrics(ticker, df_price)
        except Exception as e:
            print(f"  [ERROR] {ticker} 계산 중 오류: {e}")
            metrics = None

        if metrics is None:
            print(f"  [SKIP] {ticker}: 데이터 부족 또는 수익률 계산 실패")
            continue

        rows.append(metrics)

    if not rows:
        print("[ERROR] 유효한 RS 계산 결과가 없습니다. 가격/데이터를 확인하세요.")
        return

    df = pd.DataFrame(rows)

    # 2) 종목 RS (0~100, 1~99) 계산
    # ascending=True → 수익률 높은 종목일수록 pct/RS 값도 커짐
    pct = df["onil_weighted_ret"].rank(pct=True, ascending=True)
    df["rs_onil"] = pct * 100.0
    df["rs_onil_99"] = (pct * 98 + 1).round().astype(int)

    # ticker를 대문자로 통일
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # 3) 산업군 정보 붙이기
    df = attach_industry_info(df)

    # 4) 산업군 RS/랭크/등급 계산
    group_df = calc_group_rs(df)

    # 5) 산업군 요약 저장
    today_str = datetime.today().strftime("%Y%m%d")
    industry_out = f"industry_rs_6m_{today_str}.csv"
    group_df.to_csv(industry_out, index=False, encoding="utf-8-sig")
    print(f"[INFO] 산업군 RS 요약 저장: {industry_out}")

    # 6) 종목 레벨에 산업군 정보 merge
    df = df.merge(
        group_df[
            [
                "group_key",
                "n_members",
                "avg_ret_6m",
                "group_rank",
                "group_rs_99",
                "group_rs_100",
                "group_grade",
            ]
        ],
        on="group_key",
        how="left",
    )

    # 대시보드 호환용
    df["group_rs_6m"] = df["group_rs_100"]

    # 7) 전체 RS 결과 저장
    out_all = f"rs_onil_all_{today_str}.csv"
    df.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[INFO] 전체 RS 결과 저장: {out_all}")

    # 8) 간단 watchlist (예시: RS 상위 300개)
    watchlist = df.sort_values("rs_onil", ascending=False).head(300)
    out_watch = f"rs_onil_watchlist_{today_str}.csv"
    watchlist.to_csv(out_watch, index=False, encoding="utf-8-sig")
    print(f"[INFO] Watchlist 저장: {out_watch}")

    print("=== calc_rs_onil 완료 ===")


if __name__ == "__main__":
    main()
