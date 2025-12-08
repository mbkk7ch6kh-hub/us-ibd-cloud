# calc_rs.py

import os
from datetime import datetime

import pandas as pd


PRICES_DIR = "prices"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼 이름을 소문자 + 언더스코어로 통일
    예: 'Adj Close' -> 'adj_close'
    """
    new_cols = []
    for c in df.columns:
        name = str(c).strip().lower().replace(" ", "_")
        new_cols.append(name)
    df.columns = new_cols
    return df


def calc_1y_return_for_ticker(file_path: str, lookback_days: int = 252):
    """
    개별 종목 CSV에서 최근 1년 수익률 계산
    파일 형식: prices/{TICKER}.csv
    """
    ticker = os.path.basename(file_path).replace(".csv", "")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[{ticker}] 파일 읽기 실패: {e}")
        return None

    if df.empty:
        print(f"[{ticker}] 빈 데이터")
        return None

    df = normalize_columns(df)

    if "date" not in df.columns or "close" not in df.columns:
        print(f"[{ticker}] 'date' 또는 'close' 컬럼 없음, 건너뜀")
        return None

    # 날짜 정렬
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) <= lookback_days:
        print(f"[{ticker}] 데이터 일수 부족 ({len(df)}일) → 최소 {lookback_days+1}일 필요")
        return None

    # 최근 종가와 1년 전 종가
    last_close = df["close"].iloc[-1]
    past_close = df["close"].iloc[-(lookback_days + 1)]

    if past_close == 0 or pd.isna(last_close) or pd.isna(past_close):
        print(f"[{ticker}] 가격 데이터 이상, 건너뜀")
        return None

    ret_1y = (last_close / past_close) - 1.0
    return {
        "ticker": ticker,
        "ret_1y": ret_1y,
        "last_date": df["date"].iloc[-1],
        "last_close": last_close,
    }


def main():
    print("=== RS 계산 시작 ===")

    if not os.path.exists(PRICES_DIR):
        print(f"'{PRICES_DIR}' 폴더가 없습니다. 먼저 download_prices.py를 실행해 주세요.")
        return

    # prices 폴더 안의 모든 .csv 파일 대상
    files = [
        os.path.join(PRICES_DIR, f)
        for f in os.listdir(PRICES_DIR)
        if f.lower().endswith(".csv")
    ]

    if not files:
        print("prices 폴더 안에 CSV 파일이 없습니다.")
        return

    results = []
    for i, file_path in enumerate(files, start=1):
        ticker = os.path.basename(file_path).replace(".csv", "")
        print(f"({i}/{len(files)}) {ticker} 1년 수익률 계산 중...")
        res = calc_1y_return_for_ticker(file_path)
        if res is not None:
            results.append(res)

    if not results:
        print("유효한 결과가 없습니다. 데이터를 확인해 주세요.")
        return

    df = pd.DataFrame(results)

    # RS 등급 (퍼센타일 * 100)
    df["rs_rating"] = df["ret_1y"].rank(pct=True) * 100

    # 수익률 기준 내림차순 정렬
    df = df.sort_values("rs_rating", ascending=False).reset_index(drop=True)

    # 상위 50개만 먼저 보기 (원하면 숫자 바꿔도 됨)
    top_n = 50
    top_df = df.head(top_n)

    # CSV로 저장 (엑셀에서 바로 열 수 있음)
    today = datetime.today().strftime("%Y%m%d")
    out_file = f"rs_watchlist_{today}.csv"
    top_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n=== RS 계산 완료 ===")
    print(f"총 유효 종목 수: {len(df)}개")
    print(f"상위 {top_n}개 종목을 '{out_file}' 파일로 저장했습니다.")



if __name__ == "__main__":
    main()
