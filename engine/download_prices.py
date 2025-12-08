# download_prices.py

import os
from datetime import datetime

import pandas as pd
import yfinance as yf


UNIVERSE_FILE = "us_universe.csv"
PRICES_DIR = "prices"


def load_universe(limit: int | None = 50) -> list[str]:
    """
    us_universe.csv에서 티커 목록을 불러온다.
    limit: 처음에는 너무 많이 하지 말고 일부만 테스트 (기본 50개)
    """
    print("[DEBUG] 유니버스 파일 로드 시작:", UNIVERSE_FILE)

    if not os.path.exists(UNIVERSE_FILE):
        print(f"[ERROR] 유니버스 파일을 찾을 수 없습니다: {UNIVERSE_FILE}")
        return []

    df = pd.read_csv(UNIVERSE_FILE)

    if "symbol" not in df.columns:
        print("[ERROR] us_universe.csv 안에 'symbol' 컬럼이 없습니다. 파일 내용을 확인해 주세요.")
        print("현재 컬럼:", df.columns.tolist())
        return []

    tickers = df["symbol"].dropna().astype(str).tolist()

    if limit is not None:
        tickers = tickers[:limit]

    print(f"[DEBUG] 유니버스 로드 완료. 티커 수: {len(tickers)}개")
    return tickers


def ensure_prices_dir():
    """
    prices 폴더 없으면 생성
    """
    if not os.path.exists(PRICES_DIR):
        os.makedirs(PRICES_DIR)
        print(f"[DEBUG] '{PRICES_DIR}' 폴더 생성함.")
    else:
        print(f"[DEBUG] '{PRICES_DIR}' 폴더 이미 존재.")


def download_price_for_ticker(ticker: str, start: str = "2015-01-01") -> None:
    """
    특정 티커에 대해 yfinance에서 일봉 데이터 내려받고 prices/{ticker}.csv로 저장
    """
    print(f"[{ticker}] 다운로드 시작...")

    try:
        # yf.download 대신 Ticker().history 사용
        t = yf.Ticker(ticker)
        df = t.history(start=start, auto_adjust=True)
    except Exception as e:
        print(f"[{ticker}] 다운로드 실패: {e}")
        return

    if df is None or df.empty:
        print(f"[{ticker}] 데이터 없음 또는 빈 데이터")
        return

    # 인덱스(날짜)를 컬럼으로
    df.reset_index(inplace=True)

    # 컬럼명 통일: Date, Open, High, Low, Close, Volume → 소문자 + 언더스코어
    new_cols = []
    for c in df.columns:
        name = str(c).strip().lower().replace(" ", "_")
        new_cols.append(name)
    df.columns = new_cols

    # 디버그용 컬럼 출력
    if "date" not in df.columns or "close" not in df.columns:
        print(f"[{ticker}] 'date' 또는 'close' 컬럼 없음, 저장 생략. 현재 컬럼: {df.columns.tolist()}")
        return

    # 저장 경로
    file_path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"[{ticker}] 저장 완료 → {file_path}")


def main():
    print("=== 미국 주식 가격 데이터 다운로드 시작 ===")

    # 1) 티커 목록 로드 
    tickers = load_universe(limit=None)
    if not tickers:
        print("[ERROR] 티커 목록이 비어 있습니다. us_universe.csv를 확인해 주세요.")
        return

    # 2) prices 폴더 준비
    ensure_prices_dir()

    # 3) 각 티커별로 가격 데이터 다운로드
    for i, ticker in enumerate(tickers, start=1):
        print(f"\n({i}/{len(tickers)}) {ticker}")
        download_price_for_ticker(ticker, start="2015-01-01")

    print("\n=== 다운로드 완료 ===")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    print("[DEBUG] download_prices.py 직접 실행됨. main() 호출.")
    main()
