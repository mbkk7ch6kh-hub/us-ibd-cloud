import os
import json
import time
import datetime as dt
from typing import Dict, Any, Optional

import requests
import pandas as pd


def sec_headers() -> Dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        ua = "Your Name (your_email@example.com)"
        print("[WARN] SEC_USER_AGENT 환경변수가 비어 있습니다. (권장: 연락처 포함)")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }


def submissions_url(cik10: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik10}.json"


def load_cached(cache_path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cache(cache_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_submissions(cik10: str, cache_dir: str, sleep_sec: float = 0.15) -> Optional[Dict[str, Any]]:
    cache_path = os.path.join(cache_dir, f"CIK{cik10}.json")
    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    url = submissions_url(cik10)
    try:
        with requests.Session() as s:
            r = s.get(url, headers=sec_headers(), timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            save_cache(cache_path, data)
            time.sleep(sleep_sec)  # SEC 예절
            return data
    except Exception as e:
        print(f"[WARN] CIK {cik10} submissions fetch 실패: {e}")
        return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    engine_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(engine_dir, "cache", "submissions")
    os.makedirs(cache_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"universe.csv가 없습니다: {universe_path} (먼저 fetch_universe_sec.py 실행)")

    df = pd.read_csv(universe_path, dtype={"cik": str, "symbol": str})
    if "cik" not in df.columns or "symbol" not in df.columns:
        raise ValueError("universe.csv에 cik/symbol 컬럼이 필요합니다.")

    # 이미 sic가 있으면 재수집하지 않음
    if "sic" not in df.columns:
        df["sic"] = pd.NA
    if "sic_description" not in df.columns:
        df["sic_description"] = pd.NA

    # 결측인 것만 대상
    targets = df[df["sic"].isna()][["symbol", "cik"]].copy()
    total = len(targets)

    print(f"[STEP 1] SIC 보강 대상: {total:,} / 전체 {len(df):,}")

    sic_list = []
    sic_desc_list = []

    for i, row in enumerate(targets.itertuples(index=False), start=1):
        sym = row.symbol
        cik10 = str(row.cik).zfill(10)

        data = fetch_submissions(cik10, cache_dir=cache_dir)

        sic = None
        sic_desc = None
        if data:
            # SEC submissions JSON에 sic / sicDescription 존재하는 경우가 많음
            sic = data.get("sic")
            sic_desc = data.get("sicDescription")

        sic_list.append(sic)
        sic_desc_list.append(sic_desc)

        if i % 200 == 0 or i == total:
            print(f"  진행: {i:,}/{total:,} (예: {sym})")

    # 결과 반영
    df.loc[df["sic"].isna(), "sic"] = sic_list
    df.loc[df["sic_description"].isna(), "sic_description"] = sic_desc_list

    df["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    out_path = os.path.join(data_dir, "universe.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 간단 리포트
    filled = df["sic"].notna().sum()
    print(f"[OK] universe.csv 업데이트 완료: {out_path}")
    print(f"SIC 채움: {filled:,}/{len(df):,} ({filled/len(df)*100:.1f}%)")
    print("TIP: SIC는 '우리 산업군 RS'의 group_key로 바로 쓸 수 있음.")


if __name__ == "__main__":
    main()
