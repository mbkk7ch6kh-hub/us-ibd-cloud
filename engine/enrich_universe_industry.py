# enrich_universe_industry.py
#
# us_universe.csv에 미국 전체 티커에 대해
# sector / industry 정보를 채워 넣는 스크립트.
# - 이미 sector/industry가 채워진 종목은 건너뜀 (resume 가능)
# - 중간중간 저장하면서 진행

import time
from datetime import datetime
import os

import pandas as pd
import yfinance as yf


UNIVERSE_FILE = "us_universe.csv"


def main():
    if not os.path.exists(UNIVERSE_FILE):
        print(f"'{UNIVERSE_FILE}' 파일이 없습니다. 먼저 fetch_us_universe.py를 실행해 주세요.")
        return

    df = pd.read_csv(UNIVERSE_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    # 티커 컬럼 찾기
    if "symbol" in df.columns:
        ticker_col = "symbol"
    elif "ticker" in df.columns:
        ticker_col = "ticker"
    else:
        print(f"티커 컬럼('symbol' 또는 'ticker')을 찾을 수 없습니다. 현재 컬럼: {df.columns.tolist()}")
        return

    # 티커 컬럼 이름을 통일
    if ticker_col != "ticker":
        df = df.rename(columns={ticker_col: "ticker"})
        ticker_col = "ticker"

    # sector / industry 컬럼 없으면 생성
    if "sector" not in df.columns:
        df["sector"] = None
    if "industry" not in df.columns:
        df["industry"] = None

    # 백업 파일 한 번 저장
    backup_file = f"us_universe_backup_{datetime.today().strftime('%Y%m%d')}.csv"
    df.to_csv(backup_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] 백업 파일 저장: {backup_file}")

    # 아직 sector/industry가 비어 있는 티커만 대상
    mask_missing = df["sector"].isna() & df["industry"].isna()
    targets = df[mask_missing].copy()

    total = len(df)
    n_targets = len(targets)
    print(f"총 {total}개 티커 중, sector/industry 없는 티커: {n_targets}개")

    if n_targets == 0:
        print("추가로 채울 메타데이터가 없습니다. 종료합니다.")
        return

    # 순차적으로 yfinance로 메타데이터 조회
    for i, (idx, row) in enumerate(targets.iterrows(), start=1):
        ticker = row[ticker_col]
        print(f"({i}/{n_targets}) {ticker} 메타데이터 조회 중...")

        sector = None
        industry = None

        try:
            t = yf.Ticker(ticker)
            # 최신 yfinance에서는 get_info()가 권장, 없으면 info 사용
            try:
                info = t.get_info()
            except AttributeError:
                info = t.info

            if isinstance(info, dict):
                sector = info.get("sector")
                # industry 키 이름이 다를 수 있어서 두 개 다 시도
                industry = info.get("industry") or info.get("industryDisp")
        except Exception as e:
            print(f"  [WARN] {ticker} 메타데이터 조회 실패: {e}")

        # 결과 반영
        df.at[idx, "sector"] = sector
        df.at[idx, "industry"] = industry

        # 진행 상황 중간 저장 (예: 20개마다 한 번씩)
        if i % 20 == 0 or i == n_targets:
            df.to_csv(UNIVERSE_FILE, index=False, encoding="utf-8-sig")
            print(f"  [INFO] 진행 중 저장: {UNIVERSE_FILE} (i={i})")

        # 너무 빠르게 호출하면 yfinance / 야후가 막힐 수 있으니 약간 딜레이
        time.sleep(0.4)  # 0.4초 * 5,000티커 ≈ 33분 정도 감안

    # 최종 저장
    df.to_csv(UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    print(f"[INFO] 최종 저장 완료: {UNIVERSE_FILE}")
    print("=== 산업정보 채우기 작업 완료 ===")


if __name__ == "__main__":
    main()
