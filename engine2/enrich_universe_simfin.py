import os
import datetime as dt
import pandas as pd
import simfin as sf


def norm_ticker(x: str) -> str:
    """
    티커 정규화:
    - 대문자
    - BRK.B / BF.B 같은 클래스주를 BRK-B 형태로 맞추기
    - 슬래시 등도 '-'로 통일
    """
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(".", "-").replace("/", "-")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates_lower: list[str]) -> str | None:
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
        raise FileNotFoundError(f"universe.csv not found: {universe_path}")

    # --- SimFin 설정 ---
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
    u["symbol_key"] = u["symbol"].map(norm_ticker)

    print("[STEP1] Load SimFin companies (US) ...")
    companies = sf.load_companies(market="us")

    # ticker가 index로 올 수도 있어서 안전하게 reset
    if hasattr(companies, "index") and companies.index.name is not None:
        companies = companies.reset_index()

    c = _normalize_columns(companies)

    ticker_col = _find_col(c, ["ticker", "symbol"])
    name_col   = _find_col(c, ["company name", "name", "company"])
    sector_col = _find_col(c, ["sector", "sector name"])
    ind_col    = _find_col(c, ["industry", "industry name"])

    sector_id_col = _find_col(c, ["sectorid", "sector_id", "sector id"])
    ind_id_col    = _find_col(c, ["industryid", "industry_id", "industry id"])

    print(f"[DEBUG] companies columns = {list(c.columns)}")
    print(f"[DEBUG] ticker_col={ticker_col}, sector_col={sector_col}, ind_col={ind_col}, "
          f"sector_id_col={sector_id_col}, ind_id_col={ind_id_col}")

    if ticker_col is None:
        raise ValueError("SimFin companies에서 ticker/symbol 컬럼을 찾지 못했습니다.")

    out = pd.DataFrame()
    out["symbol"] = c[ticker_col].astype(str).str.strip()
    out["symbol_key"] = out["symbol"].map(norm_ticker)

    # 회사명(있으면)
    out["simfin_company_name"] = c[name_col].astype(str).str.strip() if name_col else pd.NA

    # --- Sector/Industry 이름이 있으면 그대로, 없으면 ID → 이름 매핑 ---
    if sector_col:
        out["sector"] = c[sector_col].astype(str).str.strip()
    else:
        out["sector"] = pd.NA

    if ind_col:
        out["industry"] = c[ind_col].astype(str).str.strip()
    else:
        out["industry"] = pd.NA

    # ID가 있고 이름이 없으면 매핑 시도
    if (out["sector"].isna().all() or out["industry"].isna().all()) and (sector_id_col or ind_id_col):
        print("[INFO] Sector/Industry name not found. Try mapping from SectorId/IndustryId...")

        if sector_id_col:
            out["sector_id"] = c[sector_id_col]
        else:
            out["sector_id"] = pd.NA

        if ind_id_col:
            out["industry_id"] = c[ind_id_col]
        else:
            out["industry_id"] = pd.NA

        # SimFin 산업/섹터 테이블 로드 (버전에 따라 함수명이 다를 수 있어 try 처리)
        sectors = None
        industries = None

        try:
            sectors = sf.load_sectors(market="us")
        except Exception as e:
            print(f"[WARN] sf.load_sectors() 실패: {e}")

        try:
            industries = sf.load_industries(market="us")
        except Exception as e:
            print(f"[WARN] sf.load_industries() 실패: {e}")

        if sectors is not None:
            sectors = _normalize_columns(sectors.reset_index() if getattr(sectors, "index", None) is not None else sectors)
            sid_col = _find_col(sectors, ["sectorid", "sector_id", "sector id", "id"])
            sname_col = _find_col(sectors, ["sector", "sector name", "name"])
            if sid_col and sname_col:
                smap = sectors[[sid_col, sname_col]].dropna().drop_duplicates()
                smap.columns = ["sector_id", "sector_name"]
                out = out.merge(smap, on="sector_id", how="left")
                out["sector"] = out["sector"].fillna(out["sector_name"])
                out = out.drop(columns=["sector_name"], errors="ignore")

        if industries is not None:
            industries = _normalize_columns(industries.reset_index() if getattr(industries, "index", None) is not None else industries)
            iid_col = _find_col(industries, ["industryid", "industry_id", "industry id", "id"])
            iname_col = _find_col(industries, ["industry", "industry name", "name"])
            if iid_col and iname_col:
                imap = industries[[iid_col, iname_col]].dropna().drop_duplicates()
                imap.columns = ["industry_id", "industry_name"]
                out = out.merge(imap, on="industry_id", how="left")
                out["industry"] = out["industry"].fillna(out["industry_name"])
                out = out.drop(columns=["industry_name"], errors="ignore")

    out = out.dropna(subset=["symbol_key"]).drop_duplicates(subset=["symbol_key"])
    print(f"[INFO] SimFin tickers(key) rows: {len(out):,}")

    print("[STEP1] Merge into universe by normalized symbol_key ...")
    merged = u.merge(
        out[["symbol_key", "sector", "industry", "simfin_company_name"]],
        on="symbol_key",
        how="left"
    )

    merged["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    merged.to_csv(universe_path, index=False, encoding="utf-8-sig")

    filled_sector = merged["sector"].notna().sum()
    filled_ind = merged["industry"].notna().sum()
    print(f"[OK] universe.csv enriched: {universe_path}")
    print(f"Sector filled: {filled_sector:,}/{len(merged):,} ({filled_sector/len(merged)*100:.1f}%)")
    print(f"Industry filled: {filled_ind:,}/{len(merged):,} ({filled_ind/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
