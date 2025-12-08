"""
download_prices.py

- us_universe.csv 에 있는 미국 종목 전체를 대상으로
- yfinance 를 이용해 '배치 단위'로 가격 데이터를 다운로드한다.
- 티커 여러 개를 한 번에 요청해서 (요청 수를 크게 줄이고),
  Too Many Requests(429) 가 뜨면 잠깐 쉬었다가 다음 배치로 진행한다.

GitHub Actions (engine/ 작업 디렉터리)에서도 그대로 동작하도록
상대 경로를 사용한다.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(BASE_DIR, "us_universe.csv")
PRICES_DIR = os.path.join(BASE_DIR, "prices")

# 한 번에 다운로드할 티커 개수 (너무 크게 잡으면 응답이 무거워질 수 있음)
BATCH_SIZE = 80

# 배치 사이에 잠깐 쉬는 시간(초)
SLEEP_BETWEEN_BATCHES = 3

# RS 계산에 필요한 건 12개월 수익률이므로,
# 여유를 두고 최근 400 거래일 정도만 받도록 설정 (대략 1년 반 수준)
DAYS_BACK = 2000


def load_universe(path: str) -> list[str]:
    """us_universe.csv 에서 티커 목록을 읽어온다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"유니버스 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("us_universe.csv 에 'symbol' 컬럼이 없습니다.")

    tickers = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"[INFO] 유니버스 티커 수: {len(tickers)}개")
    return tickers


def ensure_prices_dir():
    if not os.path.exists(PRICES_DIR):
        os.makedirs(PRICES_DIR, exist_ok=True)
        print(f"[INFO] prices 폴더 생성: {PRICES_DIR}")


def yf_download_batch(tickers_batch, start_date: datetime) -> pd.DataFrame | None:
    """
    yfinance.download 로 여러 티커를 한 번에 다운로드.
    - 성공하면 DataFrame 반환 (단일/다중 컬럼 모두 가능)
    - 실패하면 None 반환
    """
    if not tickers_batch:
        return None

    tickers_str = " ".join(tickers_batch)
    print(f"[BATCH] {tickers_batch[0]} ... {tickers_batch[-1]} ({len(tickers_batch)}개) 다운로드 요청")

    try:
        data = yf.download(
            tickers=tickers_str,
            start=start_date.strftime("%Y-%m-%d"),
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        return data
    except Exception as e:
        print(f"[BATCH ERROR] {tickers_batch[0]} ... {tickers_batch[-1]} 배치 다운로드 실패: {e}")
        return None


def save_single_ticker_from_batch(ticker: str, batch_df: pd.DataFrame):
    """
    배치 다운로드 결과(batch_df)에서 특정 ticker의 시계열만 꺼내서
    prices/{ticker}.csv 로 저장한다.
    """
    # 배치 결과 컬럼 형태에 따라 분기
    if isinstance(batch_df.columns, pd.MultiIndex):
        # (ticker, field) 형태
        # 예: ('AAPL', 'Close')
        if (ticker, "Close") not in batch_df.columns:
            print(f"  [SKIP] {ticker}: 배치 내에 데이터 컬럼 없음")
            return

        df_t = batch_df.xs(ticker, axis=1, level=0).copy()
    else:
        # 단일 티커만 호출한 경우 (DataFrame with columns ['Open','High',...])
        df_t = batch_df.copy()

    if df_t.empty:
        print(f"  [SKIP] {ticker}: 데이터 프레임이 비어 있음")
        return

    # 인덱스(Date)를 컬럼으로
    df_t = df_t.reset_index()

    # 컬럼 이름 통일
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df_t = df_t.rename(columns=rename_map)

    # 최소한 date, close 가 있어야 의미 있음
    if "date" not in df_t.columns or "close" not in df_t.columns:
        print(f"  [SKIP] {ticker}: 'date' 또는 'close' 컬럼 없음")
        return

    # 정렬
    df_t = df_t.sort_values("date")

    out_path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    df_t.to_csv(out_path, index=False)
    print(f"  [SAVE] {ticker}: {len(df_t)}개 레코드 저장 -> {out_path}")


def main():
    print("=== 미국 주식 가격 데이터 다운로드 시작 (배치 버전) ===")

    ensure_prices_dir()

    try:
        tickers = load_universe(UNIVERSE_FILE)
    except Exception as e:
        print("Error:  유니버스 파일을 찾을 수 없습니다 또는 형식 오류:", e)
        return

    if not tickers:
        print("Error:  티커 목록이 비어 있습니다. us_universe.csv를 확인해 주세요.")
        return

    # RS 계산에 필요한 날짜 범위
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=DAYS_BACK)
    print(f"[INFO] 가격 데이터 기간: {start_date} ~ {today} (약 {DAYS_BACK}일 전부터)")

    total = len(tickers)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(num_batches):
        start_idx = b * BATCH_SIZE
        end_idx = min((b + 1) * BATCH_SIZE, total)
        batch = tickers[start_idx:end_idx]

        print(f"\n=== 배치 {b + 1}/{num_batches} ({start_idx + 1} ~ {end_idx}번째 티커) ===")

        batch_df = yf_download_batch(batch, start_date)
        if batch_df is None or batch_df.empty:
            print("[WARN] 배치 전체가 비어 있거나 다운로드 실패. 다음 배치로 이동.")
            # 429 등으로 막혔을 가능성도 있으니 조금 더 쉬었다 간다.
            time.sleep(10)
            continue

        # MultiIndex(여러 티커) / 단일티커 모두 처리 가능
        for tkr in batch:
            try:
                save_single_ticker_from_batch(tkr, batch_df)
            except Exception as e:
                print(f"  [SKIP] {tkr}: 개별 저장 중 오류: {e}")

        # 배치 간 간격을 둬서 과도한 요청 방지
        print(f"[INFO] 배치 {b + 1}/{num_batches} 처리 완료. {SLEEP_BETWEEN_BATCHES}초 대기 후 다음 배치.")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("=== 미국 주식 가격 데이터 다운로드 완료 (배치 버전) ===")


if __name__ == "__main__":
    print("[DEBUG] download_prices.py 직접 실행됨. main() 호출.")
    main()
