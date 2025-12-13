import os
import time
import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf


UNIVERSE_FILE = "us_universe.csv"
OUT_DIR = "."
PRICE_PERIOD = "500d"  # ~ 1.5년, 12개월 수익률 계산에 충분
CHUNK_SIZE = 200       # 한번에 요청할 티커 수 (전체가 아니라 요청 단위일 뿐)
SLEEP_SEC = 1.0        # 청크 사이 딜레이 (레이트 리밋 완화용)


def load_universe(path: str) -> pd.DataFrame:
    """
    us_universe.csv에서 symbol, group_key를 읽어온다.
    - symbol / ticker / Symbol / Ticker / 종목코드 등 여러 이름을 허용
    - group_key가 없으면 sector/industry 조합으로 생성
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"유니버스 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)

    # 티커 컬럼 찾기
    symbol_col = None
    for c in ["symbol", "ticker", "Symbol", "Ticker", "종목코드"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        raise ValueError(
            f"유니버스 파일에서 티커 컬럼(symbol/ticker)을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}"
        )

    # 내부적으로는 'symbol'로 통일
    if symbol_col != "symbol":
        df = df.rename(columns={symbol_col: "symbol"})

    # group_key 없으면 sector/industry 조합으로 생성
    if "group_key" not in df.columns:
        sector_col = None
        industry_col = None
        for c in ["sector", "Sector", "섹터"]:
            if c in df.columns:
                sector_col = c
                break
        for c in ["industry", "Industry", "산업", "industry_group"]:
            if c in df.columns:
                industry_col = c
                break

        if sector_col is not None and industry_col is not None:
            df["group_key"] = (
                df[sector_col].fillna("Unknown") + " | " + df[industry_col].fillna("Unknown")
            )
        elif sector_col is not None:
            df["group_key"] = df[sector_col].fillna("Unknown")
        elif industry_col is not None:
            df["group_key"] = df[industry_col].fillna("Unknown")
        else:
            df["group_key"] = "Unknown"

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["symbol"])
    df = df[df["symbol"] != ""]

    return df[["symbol", "group_key"]].drop_duplicates()


def download_prices_chunk(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    yfinance로 여러 티커를 한 번에 다운로드.
    반환값: {티커: DataFrame(OHLCV)}
    """
    if not tickers:
        return {}

    tickers_str = " ".join(tickers)
    # auto_adjust=True 로 분할/배당 반영된 Adjusted price 사용
    data = yf.download(
        tickers=tickers_str,
        period=PRICE_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    result: Dict[str, pd.DataFrame] = {}

    # MultiIndex 인지 여부 체크
    if isinstance(data.columns, pd.MultiIndex):
        # data[ticker] 가 각 티커별 서브프레임
        for t in tickers:
            if t in data.columns.get_level_values(0):
                try:
                    px = data[t].copy()
                    # 칼럼 이름 정리
                    px = px.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Adj Close": "close",
                            "Volume": "volume",
                        }
                    )
                    px = px.reset_index().rename(columns={"Date": "date"})
                    result[t] = px
                except Exception:
                    continue
    else:
        # 단일 티커일 때
        t = tickers[0]
        px = data.copy()
        px = px.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "close",
                "Volume": "volume",
            }
        )
        px = px.reset_index().rename(columns={"Date": "date"})
        result[t] = px

    return result


def calc_returns_from_prices(px: pd.DataFrame):
    """
    px: columns = [date, open, high, low, close, volume]
    3, 6, 9, 12개월 수익률과 보조 지표 계산
    """
    if "date" not in px.columns or "close" not in px.columns:
        return None

    # 날짜 정렬
    px = px.sort_values("date").reset_index(drop=True)
    if px["close"].notna().sum() < 80:  # 최소 3개월은 있어야 의미 있음
        return None

    last_idx = px.index[px["close"].last_valid_index()]
    last_close = float(px.loc[last_idx, "close"])
    last_date = pd.to_datetime(px.loc[last_idx, "date"]).date()

    def ret_n(days: int):
        if last_idx - days < 0:
            return np.nan
        base = px.loc[last_idx - days, "close"]
        if pd.isna(base) or base == 0:
            return np.nan
        return float(last_close / base - 1.0)

    # 대략적 거래일 기준 (3,6,9,12개월)
    ret_3m = ret_n(63)
    ret_6m = ret_n(126)
    ret_9m = ret_n(189)
    ret_12m = ret_n(252)

    # 거래량 지표
    vol = px["volume"]
    vol = pd.to_numeric(vol, errors="coerce")
    avg_vol_50 = float(vol.tail(50).mean()) if vol.notna().sum() > 0 else np.nan
    avg_dollar_vol_50 = float(avg_vol_50 * last_close) if not np.isnan(avg_vol_50) else np.nan

    # 오닐식 가중 수익률 (12m*3 + 9m*2 + 6m + 3m) / 7
    weights = []
    vals = []
    for r, w in [(ret_12m, 3), (ret_9m, 2), (ret_6m, 1), (ret_3m, 1)]:
        if not np.isnan(r):
            vals.append(r * w)
            weights.append(w)

    if len(weights) == 0:
        weighted_ret = np.nan
    else:
        weighted_ret = float(sum(vals) / sum(weights))

    return {
        "last_date": last_date,
        "last_close": last_close,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_9m": ret_9m,
        "ret_12m": ret_12m,
        "onil_weighted_ret": weighted_ret,
        "avg_vol_50": avg_vol_50,
        "avg_dollar_vol_50": avg_dollar_vol_50,
    }


def rs_scale(series: pd.Series, max_score: int = 99) -> pd.Series:
    """
    백분위 순위 → 0~max_score 점수로 변환
    """
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)

    return series.rank(pct=True) * max_score


def build_industry_rs(df: pd.DataFrame) -> pd.DataFrame:
    """
    종목 RS 결과에 group_key를 붙여 산업군 RS/랭크/등급 계산
    """
    if "group_key" not in df.columns:
        df["group_key"] = "Unknown"

    grp = df.groupby("group_key", dropna=False)

    ind = grp.agg(
        n_members=("symbol", "count"),
        avg_ret_6m=("ret_6m", "mean"),
        avg_weighted_ret=("onil_weighted_ret", "mean"),
    ).reset_index()

    # 산업군 RS는 가중 수익률 기반 백분위
    ind["group_rs_100"] = ind["avg_weighted_ret"].rank(pct=True) * 100
    ind["group_rs_99"] = ind["avg_weighted_ret"].rank(pct=True) * 99

    # 랭크 (높은 RS가 1등)
    ind = ind.sort_values("group_rs_100", ascending=False)
    ind["group_rank"] = range(1, len(ind) + 1)

    # 등급 A~E
    def grade_func(score):
        if np.isnan(score):
            return "E"
        if score >= 90:
            return "A"
        if score >= 70:
            return "B"
        if score >= 40:
            return "C"
        if score >= 20:
            return "D"
        return "E"

    ind["group_grade"] = ind["group_rs_100"].apply(grade_func)
    ind = ind.rename(columns={"avg_weighted_ret": "group_rs_6m"})  # 호환성 유지

    return ind


def main():
    today = dt.date.today().strftime("%Y%m%d")
    out_rs = os.path.join(OUT_DIR, f"rs_onil_all_{today}.csv")
    out_ind = os.path.join(OUT_DIR, f"industry_rs_6m_{today}.csv")

    print("=== US IBD RS (O'Neil style) 계산 시작 ===")
    print(f"[INFO] 유니버스 파일: {UNIVERSE_FILE}")

    uni = load_universe(UNIVERSE_FILE)
    tickers = uni["symbol"].tolist()
    print(f"[INFO] 유니버스 티커 수: {len(tickers)}")

    records = []
    total = len(tickers)

    for i in range(0, total, CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        print(f"[CHUNK] {i+1} ~ {min(i+CHUNK_SIZE, total)} 티커 다운로드 중...")
        try:
            prices_dict = download_prices_chunk(chunk)
        except Exception as e:
            print(f"[ERROR] 청크 다운로드 실패: {e}")
            time.sleep(SLEEP_SEC)
            continue

        for t in chunk:
            px = prices_dict.get(t)
            if px is None or px.empty:
                print(f"  [SKIP] {t}: 가격 데이터 없음 또는 형식 오류")
                continue

            metrics = calc_returns_from_prices(px)
            if metrics is None:
                print(f"  [SKIP] {t}: 수익률 계산 불가 (데이터 부족)")
                continue

            rec = {"symbol": t}
            rec.update(metrics)
            # group_key 붙이기
            g = uni.loc[uni["symbol"] == t, "group_key"]
            rec["group_key"] = g.iloc[0] if not g.empty else "Unknown"
            records.append(rec)

        # 레이트 리밋 완화
        time.sleep(SLEEP_SEC)

    if not records:
        print("[ERROR] 유효한 RS 계산 결과가 없습니다. 가격/데이터를 확인하세요.")
        return

    df = pd.DataFrame(records)
    print(f"[INFO] RS 계산 완료 티커 수: {len(df)}")

    # RS 점수 (0~99)
    df["rs_onil_99"] = rs_scale(df["onil_weighted_ret"], max_score=99)
    df["rs_onil"] = df["rs_onil_99"]

    # 산업군 RS 계산
    ind_df = build_industry_rs(df)

    # 산업군 정보 merge
    df = df.merge(
        ind_df[
            [
                "group_key",
                "n_members",
                "avg_ret_6m",
                "group_rs_6m",
                "group_rs_99",
                "group_rs_100",
                "group_rank",
                "group_grade",
            ]
        ],
        on="group_key",
        how="left",
    )

    # 컬럼 정렬 (기존 파일과 최대한 호환)
    cols = [
        "symbol",
        "last_date",
        "last_close",
        "ret_3m",
        "ret_6m",
        "ret_9m",
        "ret_12m",
        "onil_weighted_ret",
        "avg_vol_50",
        "avg_dollar_vol_50",
        "rs_onil",
        "rs_onil_99",
        "group_key",
        "n_members",
        "avg_ret_6m",
        "group_rs_6m",
        "group_rs_99",
        "group_rs_100",
        "group_rank",
        "group_grade",
    ]
    df = df[cols]

    df.to_csv(out_rs, index=False, encoding="utf-8-sig")
    print(f"[INFO] 종목 RS 파일 저장 완료: {out_rs}")

    # 산업군 요약은 별도 파일로 저장
    ind_df.to_csv(out_ind, index=False, encoding="utf-8-sig")
    print(f"[INFO] 산업군 RS 파일 저장 완료: {out_ind}")
    print("=== US IBD RS (O'Neil style) 계산 종료 ===")


if __name__ == "__main__":
    main()
