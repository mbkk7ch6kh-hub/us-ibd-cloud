import os
import time
import pandas as pd

import yfinance as yf


def norm_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(".", "-").replace("/", "-")
    return s


def safe_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def fetch_yahoo_sector_industry(ticker: str, retry=2, sleep_sec=0.6):
    """
    Yahoo(비공식) 특성상 종종 None이 나오거나 느리거나 차단될 수 있음.
    그래서 retry + sleep을 둠.
    """
    last_err = None
    for _ in range(retry + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.get_info()  # yfinance 버전에 따라 .info 대신 get_info()가 안정적인 편
            sector = safe_str(info.get("sector"))
            industry = safe_str(info.get("industry"))
            return sector, industry, None
        except Exception as e:
            last_err = str(e)
            time.sleep(sleep_sec)
    return None, None, last_err


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"universe.csv not found: {universe_path}")

    cache_path = os.path.join(data_dir, "yahoo_industry_cache.csv")

    # 한 번에 너무 많이 치면 차단 확률↑ → 기본 250개만 처리하고, 매일 누적 채움
    max_per_run = int(os.getenv("YAHOO_MAX_PER_RUN", "250"))
    sleep_sec = float(os.getenv("YAHOO_SLEEP_SEC", "0.6"))

    u = pd.read_csv(universe_path, dtype=str)
    u["symbol"] = u["symbol"].astype(str).str.strip()
    u["symbol_key"] = u["symbol"].map(norm_ticker)

    # group_key가 비어있는 애들만 대상
    def is_missing_group(x):
        if x is None:
            return True
        s = str(x).strip()
        return (s == "") or (s.lower() == "nan")

    if "group_key" not in u.columns:
        u["group_key"] = pd.NA

    missing_mask = u["group_key"].apply(is_missing_group)
    targets = u.loc[missing_mask, "symbol"].tolist()

    print(f"[YH] missing group_key tickers: {len(targets):,}")

    # 캐시 로드
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype=str)
        cache["symbol"] = cache["symbol"].astype(str).str.strip()
    else:
        cache = pd.DataFrame(columns=["symbol", "yahoo_sector", "yahoo_industry", "yahoo_group_key"])

    cached_set = set(cache["symbol"].tolist())

    # 아직 캐시에 없는 것만
    todo = [t for t in targets if t not in cached_set]
    print(f"[YH] to fetch this run (before cap): {len(todo):,}")

    todo = todo[:max_per_run]
    print(f"[YH] capped to: {len(todo):,} (YAHOO_MAX_PER_RUN={max_per_run})")

    new_rows = []
    for i, ticker in enumerate(todo, 1):
        sector, industry, err = fetch_yahoo_sector_industry(ticker, retry=2, sleep_sec=sleep_sec)
        if sector or industry:
            group_key = f"YH|{sector or 'Unknown'}|{industry or 'Unknown'}"
            new_rows.append([ticker, sector, industry, group_key])
            print(f"  [{i}/{len(todo)}] {ticker}: OK ({sector} / {industry})")
        else:
            # 실패도 캐시에 남겨서 같은 날 무한 재시도 방지(다음날 재도전하려면 캐시 정리 정책을 둘 수도 있음)
            new_rows.append([ticker, None, None, None])
            print(f"  [{i}/{len(todo)}] {ticker}: FAIL ({err})")

        time.sleep(sleep_sec)

    if new_rows:
        add_df = pd.DataFrame(new_rows, columns=["symbol", "yahoo_sector", "yahoo_industry", "yahoo_group_key"])
        cache = pd.concat([cache, add_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=["symbol"], keep="last")
        cache.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[YH] cache updated: {cache_path}")

    # universe에 yahoo_group_key로 보충(비어있는 group_key만)
    cache2 = cache.dropna(subset=["yahoo_group_key"]).copy()
    if len(cache2) > 0:
        u = u.merge(cache2[["symbol", "yahoo_sector", "yahoo_industry", "yahoo_group_key"]], on="symbol", how="left")
        fill_mask = u["group_key"].apply(is_missing_group) & u["yahoo_group_key"].notna()
        u.loc[fill_mask, "group_key"] = u.loc[fill_mask, "yahoo_group_key"]

    u.to_csv(universe_path, index=False, encoding="utf-8-sig")
    print("[YH] universe.csv updated with Yahoo fallback.")


if __name__ == "__main__":
    main()
