import os
import re
import datetime as dt
import pandas as pd
import simfin as sf


def norm_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(".", "-").replace("/", "-")
    return s


def norm_company_name(x: str) -> str:
    """
    회사명 정규화(우선주/예탁/시리즈/단위 같은 장식 제거)
    - 완벽하진 않지만 Unknown을 크게 줄이는 실용적 휴리스틱
    """
    if x is None:
        return ""
    s = str(x).upper()

    # 흔한 장식어 제거
    drop_words = [
        "PREFERRED", "PREF", "SERIES", "DEPOSITARY", "DEPOSITORY",
        "SHARES", "SHS", "COMMON", "CL A", "CL B", "CLASS A", "CLASS B",
        "ADR", "SPONSORED", "UNSPONSORED",
        "TRUST", "FUND",
        "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "SA", "AG", "NV",
    ]
    for w in drop_words:
        s = s.replace(w, " ")

    # 괄호/기호 제거
    s = re.sub(r"[\(\)\[\]\{\}\.,:&/\\\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
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
    u["company_name"] = u.get("company_name", "").astype(str)
    u["company_key"] = u["company_name"].map(norm_company_name)

    print("[STEP1] Load SimFin companies (US) ...")
    companies = sf.load_companies(market="us")
    if hasattr(companies, "index") and companies.index.name is not None:
        companies = companies.reset_index()
    c = _normalize_columns(companies)

    related_cols = [col for col in c.columns if ("sector" in col.lower()) or ("industry" in col.lower())]
    print(f"[DEBUG] companies total cols={len(c.columns)}")
    print(f"[DEBUG] sector/industry related cols={related_cols}")

    ticker_col = _find_col(c, ["ticker", "symbol"])
    if ticker_col is None:
        raise ValueError(f"SimFin companies에서 ticker/symbol 컬럼을 못 찾음. cols={list(c.columns)}")

    name_col = _find_col(c, ["company name", "name", "company"])
    industry_id_col = _find_col(c, ["industryid", "industry_id", "industry id"])
    sector_id_col = _find_col(c, ["sectorid", "sector_id", "sector id"])

    sector_name_col = _find_col(c, ["sector", "sector name"])
    industry_name_col = _find_col(c, ["industry", "industry name"])

    print(f"[DEBUG] ticker_col={ticker_col}, industry_id_col={industry_id_col}, sector_id_col={sector_id_col}")

    out = pd.DataFrame()
    out["symbol_key"] = c[ticker_col].astype(str).str.strip().map(norm_ticker)

    out["simfin_company_name"] = c[name_col].astype(str).str.strip() if name_col else pd.NA
    out["company_key_simfin"] = out["simfin_company_name"].map(norm_company_name) if name_col else pd.NA

    out["industry_id"] = c[industry_id_col] if industry_id_col else pd.NA
    out["sector_id"] = c[sector_id_col] if sector_id_col else pd.NA

    # 이름은 어차피 0%라 유지하되 NA 허용
    out["sector"] = c[sector_name_col].astype(str).str.strip() if sector_name_col else pd.NA
    out["industry"] = c[industry_name_col].astype(str).str.strip() if industry_name_col else pd.NA

    out["group_key"] = out["industry_id"].fillna(out["sector_id"]).astype("string")
    out = out.dropna(subset=["symbol_key"]).drop_duplicates(subset=["symbol_key"])

    print(f"[INFO] SimFin tickers(key) rows: {len(out):,}")
    print(f"[INFO] industry_id filled (SimFin side): {out['industry_id'].notna().sum():,}")

    print("[STEP1] Merge into universe by symbol_key ...")
    merged = u.merge(
        out[["symbol_key", "group_key", "industry_id", "sector_id", "sector", "industry", "simfin_company_name", "company_key_simfin"]],
        on="symbol_key",
        how="left"
    )

    # --- 핵심: 회사명 기반 보정(보통주/우선주 동시 커버) ---
    # 1) universe 내부에서 이미 group_key가 있는 종목들의 company_key → 대표 group_key 생성
    base = merged.copy()
    base_has = base[base["group_key"].notna() & (base["group_key"].astype(str).str.len() > 0)].copy()

    if len(base_has) > 0:
        # 같은 회사명에 여러 group_key가 섞이면 최빈값(mode) 사용
        company_to_group = (
            base_has.groupby("company_key")["group_key"]
            .agg(lambda s: s.value_counts().index[0])
            .reset_index()
            .rename(columns={"group_key": "group_key_from_company"})
        )

        merged = merged.merge(company_to_group, on="company_key", how="left")

        # group_key가 비어있으면 회사 기반 값으로 채움
        merged["group_key"] = merged["group_key"].fillna(merged["group_key_from_company"])
        merged = merged.drop(columns=["group_key_from_company"], errors="ignore")

    merged["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    merged.to_csv(universe_path, index=False, encoding="utf-8-sig")

    filled_group = merged["group_key"].notna().sum()
    filled_ind_id = merged["industry_id"].notna().sum()
    print(f"[OK] universe.csv enriched: {universe_path}")
    print(f"group_key filled: {filled_group:,}/{len(merged):,} ({filled_group/len(merged)*100:.1f}%)")
    print(f"industry_id filled (direct): {filled_ind_id:,}/{len(merged):,} ({filled_ind_id/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
