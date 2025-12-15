import os
import time
import datetime as dt
from typing import Dict, Any, Optional

import requests
import pandas as pd


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def sec_headers() -> Dict[str, str]:
    """
    SEC는 User-Agent(연락처 포함) + Referer 등을 요구/선호.
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "example.com" in ua or "Your Name" in ua:
        raise RuntimeError(
            "SEC_USER_AGENT 환경변수가 비어 있거나 예시값입니다.\n"
            'PowerShell에서 예: $env:SEC_USER_AGENT="US-IBD-Engine/1.0 (you@email.com)"'
        )

    return {
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sec.gov/",
        "Origin": "https://www.sec.gov",
        "Connection": "keep-alive",
    }


def fetch_json_with_retry(url: str, tries: int = 5, base_sleep: float = 1.5) -> Any:
    last_err: Optional[Exception] = None
    with requests.Session() as s:
        for i in range(1, tries + 1):
            try:
                r = s.get(url, headers=sec_headers(), timeout=60)
                if r.status_code == 403:
                    # 403은 잠깐 기다렸다 재시도해도 풀리는 경우가 있음
                    sleep_sec = base_sleep * i
                    print(f"[WARN] 403 Forbidden. 재시도 {i}/{tries} (sleep {sleep_sec:.1f}s)")
                    time.sleep(sleep_sec)
                    continue
                r.raise_for_status()
                time.sleep(0.3)  # SEC 예절
                return r.json()
            except Exception as e:
                last_err = e
                sleep_sec = base_sleep * i
                print(f"[WARN] 요청 실패: {type(e).__name__}: {e}  (sleep {sleep_sec:.1f}s)")
                time.sleep(sleep_sec)

    raise RuntimeError(f"SEC 요청 실패(최대 재시도 초과). 마지막 에러: {last_err}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("[STEP 1] SEC company_tickers.json 다운로드...")

    # 디버그: 실제로 어떤 UA로 나가는지 확인
    print(f"[DEBUG] SEC_USER_AGENT = {os.getenv('SEC_USER_AGENT')}")

    raw = fetch_json_with_retry(SEC_TICKER_URL)

    rows = []
    for _, v in raw.items():
        cik = str(v.get("cik_str", "")).strip()
        ticker = str(v.get("ticker", "")).strip()
        title = str(v.get("title", "")).strip()

        if not ticker or not cik:
            continue

        rows.append(
            {
                "symbol": ticker,
                "cik": cik.zfill(10),
                "company_name": title,
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).sort_values("symbol")
    df["updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    out_csv = os.path.join(data_dir, "universe.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] universe.csv 생성 완료: {out_csv}")
    print(f"총 종목 수: {len(df):,}")


if __name__ == "__main__":
    main()
