import os
import datetime as dt
import pandas as pd
import simfin as sf


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 strip하고, 중복/공백 이슈를 줄인다."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates_lower: list[str]) -> str | None:
    """컬럼명을 소문자 비교로 찾아준다."""
    cols = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for cand in candidates_lower:
        if cand in cols:
            return cols[cand]
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(
            f"universe.csv not found: {universe_path} (먼저 fetch_universe_nasdaqtrader.py 실행)"
        )

    # SimFin 설정
    api_key = os.getenv("SIMFIN_API_KEY", "").strip() or "free"
    sf.set_api_key(api_key)

    cache_dir = os.path.join(root, "engine2", "cache", "simfin")
    os.makedirs(cache_dir, exist_ok=True)
    sf.set_data_dir(cache_dir)

    print("[STEP1] Load universe.csv ...")
    u = pd.read_csv(universe_path, dtype=str)
    if "symbol" not in u.columns:
        raise ValueError(f"universe.csv에 symbol 컬럼이 없습니다. 현재 컬럼: {list(u.columns)}")
    u["symbol"] = u["symbol"].astype(str).str.strip()

    print("[STEP1] Load SimFin companies (US) ...")
    companies = sf.load_companies(market="us")

    # companies가 ticker를 index로 들고 올 수도 있으니 reset_index()로 안전하게 처리
    if hasattr(companies, "index") and companies.index.name is not None:
        # 예: index.name == 'Ticker' 같은 경우
        companies = companies.reset_index()

    c = _normalize_columns(companies)

    # 컬럼 후보군(버전별 차이를 흡수)
    ticker_col = _find_col(c, ["ticker", "tkr", "symbol"])
    name_col   = _find_col(c, ["company name", "name", "company"])
    sector_col = _find_col(c, ["sector", "sector name"])
    ind_col    = _find_col(c, ["industry", "industry name"])

    if ticker_col is None:
        # 디버깅을 위해 컬럼을 출력해서 바로 원인 파악 가능하게
        raise ValueError(
            "SimFin companies에서 ticker/symbol 컬럼을 찾지 못했습니다.\n"
            f"현재 companies 컬럼: {list(c.columns)}"
        )

    out = pd.DataFrame()
    out["symbol"] = c[ticker_col].astype(str).str.strip()

    if name_col is not None:
        out["simfin_company_name"] = c[name_col].astype(str).str.strip()
    else:
        out["simfin_company_name"] = pd.NA

    if sector_col is not None:
        out["sector"] = c[sector_col].astype(str).str.strip()
    else:
        out["sector"] = pd.NA

    if ind_col is not None:
        out["industry"] = c[ind_col].astype(str).str.strip()
    else:
        out["industry"] = pd.NA

    out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])

    print(f"[INFO] SimFin tickers: {len(out):,}")
    print("[STEP1] Merge sector/industry into universe ...")
    merged = u.merge(out, on="symbol", how="left")

    merged["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    merged.to_csv(universe_path, index=False, encoding="utf-8-sig")

    filled_sector = merged["sector"].notna().sum() if "sector" in merged.columns else 0
    filled_ind = merged["industry"].notna().sum() if "industry" in merged.columns else 0

    print(f"[OK] universe.csv enriched: {universe_path}")
    print(f"Sector filled: {filled_sector:,}/{len(merged):,} ({filled_sector/len(merged)*100:.1f}%)")
    print(f"Industry filled: {filled_ind:,}/{len(merged):,} ({filled_ind/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
