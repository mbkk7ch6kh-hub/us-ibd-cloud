import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
from typing import Tuple
import textwrap

# ==============================
# 기본 설정
# ==============================
st.set_page_config(
    page_title="US IBD RS Online Dashboard 🔐",
    layout="wide",
)

# ==============================
# 비밀번호 보호
# ==============================

def password_entered() -> None:
    """비밀번호 입력 후 호출되는 콜백."""
    if st.session_state.get("password", "") == st.secrets["APP_PASSWORD"]:
        st.session_state["password_correct"] = True
        st.session_state["password"] = ""  # 입력창 비우기
    else:
        st.session_state["password_correct"] = False


def check_password() -> bool:
    """비밀번호가 맞으면 True, 아니면 로그인 화면만 보여주고 False."""
    if st.session_state.get("password_correct", False):
        return True

    st.title("US IBD RS Online Dashboard 🔐")
    st.write("접근을 위해 비밀번호를 입력해 주세요.")

    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if st.session_state.get("password_correct") is False:
        st.error("비밀번호가 올바르지 않습니다.")

    return False


# ==============================
# 헬퍼 함수들
# ==============================

def normalize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS 원본 파일에서 'ticker' 컬럼이 없을 경우,
    'symbol' 등 다른 후보 컬럼을 'ticker'로 통일한다.
    """
    df = df.copy()

    if "ticker" in df.columns:
        return df

    if "symbol" in df.columns:
        df.rename(columns={"symbol": "ticker"}, inplace=True)
        return df

    # 필요시 다른 후보를 추가할 수 있음
    candidates = [c for c in ["secid", "종목코드"] if c in df.columns]
    if candidates:
        df.rename(columns={candidates[0]: "ticker"}, inplace=True)
        return df

    raise ValueError(
        f"티커 컬럼(ticker/symbol)을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}"
    )


@st.cache_data(show_spinner=False)
def load_rs_from_cloud() -> pd.DataFrame:
    """
    GitHub(data/latest_rs_smr.csv)에서 RS+SMR 데이터를 읽어온다.
    - st.secrets["RS_URL"] 을 사용
    - symbol → ticker 정규화
    """
    rs_url = st.secrets["RS_URL"]
    df = pd.read_csv(rs_url)

    # 티커 컬럼 정규화
    df = normalize_ticker_column(df)

    # 필수 컬럼 체크 (ticker 기준)
    required_cols = {
        "ticker",
        "last_date",
        "last_close",
        "ret_3m",
        "ret_6m",
        "ret_9m",
        "ret_12m",
        "rs_onil_99",
        "group_key",
        "group_rank",
        "group_rs_99",
        "group_grade",
        "smr_score",
        "smr_grade",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"필수 컬럼 {missing} 이(가) 없습니다. calc_rs_onil.py + enrich_smr.py 결과를 확인해 주세요. "
            f"현재 컬럼: {list(df.columns)}"
        )

    # 정리용 컬럼들 추가/정렬
    # 숫자형으로 한번 더 캐스팅
    num_cols = [
        "last_close",
        "ret_3m",
        "ret_6m",
        "ret_9m",
        "ret_12m",
        "onil_weighted_ret" if "onil_weighted_ret" in df.columns else None,
        "avg_vol_50",
        "avg_dollar_vol_50",
        "rs_onil" if "rs_onil" in df.columns else None,
        "rs_onil_99",
        "group_rs_99",
        "group_rs_100" if "group_rs_100" in df.columns else None,
        "group_rs_6m" if "group_rs_6m" in df.columns else None,
        "sales_growth",
        "profit_margin",
        "roe",
        "smr_score",
    ]
    num_cols = [c for c in num_cols if c is not None and c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 날짜는 문자열로 들어오므로, 최근 날짜를 상단에 보이도록 정렬할 때만 사용
    # 여기서는 last_date를 그대로 두되, 필요할 때만 parse
    return df


@st.cache_data(show_spinner=False)
def load_industry_from_cloud() -> pd.DataFrame:
    """
    GitHub(data/latest_industry_rs.csv)에서 산업군 RS 데이터를 읽어온다.
    - st.secrets["INDUSTRY_URL"] 을 사용
    """
    ind_url = st.secrets["INDUSTRY_URL"]
    try:
        df = pd.read_csv(ind_url)
    except Exception as e:
        st.warning(f"산업군 RS 데이터를 불러오는 중 문제가 발생했습니다: {e}")
        return pd.DataFrame()

    # group_key가 주요 키
    if "group_key" not in df.columns:
        st.warning("industry_rs 파일에 'group_key' 컬럼이 없습니다.")
        return pd.DataFrame()

    # 숫자형 캐스팅
    for c in ["group_rank", "group_rs_99", "group_rs_100", "avg_ret_6m", "n_members"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def format_percentage(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x*100:,.1f}%"


def format_price(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.2f}"


def short_k(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:.0f}"


def tradingview_embed_symbol(ticker: str) -> str:
    """
    TradingView 임베드용 심볼 문자열.
    - 미국 주식 기본 가정: 'NASDAQ:TSLA' 형식. (거래소 정보가 없다면 기본은 'NYSE 또는 NASDAQ' 추정이 어려우니,
      심볼만으로도 동작하는 위젯 URL을 사용)
    """
    # 단순하게 심볼만 쓰는 형태의 TV URL로 구성
    return ticker.upper()


def render_tradingview_chart(ticker: str):
    """
    선택한 ticker에 대해 TradingView 위젯을 iframe으로 임베드.
    """
    import streamlit.components.v1 as components

    symbol = tradingview_embed_symbol(ticker)
    # TV 위젯 URL (심플 차트)
    # 필요하면 interval, theme, studies 등을 나중에 조정 가능
    tv_url = (
        "https://s.tradingview.com/widgetembed/"
        "?symbol={symbol}"
        "&interval=D"
        "&hidesidetoolbar=1"
        "&symboledit=1"
        "&saveimage=0"
        "&toolbarbg=f1f3f6"
        "&studies=[]"
        "&theme=light"
        "&style=1"
        "&timezone=exchange"
        "&withdateranges=1"
        "&hideideas=1"
        "&enable_publishing=0"
        "&allow_symbol_change=1"
    ).format(symbol=symbol)

    components.iframe(tv_url, height=600, scrolling=False)


# ==============================
# 메인 앱
# ==============================

def main():
    if not check_password():
        st.stop()

    st.title("US IBD RS Online Dashboard")

    # 데이터 로드
    with st.spinner("RS + SMR 데이터를 불러오는 중입니다..."):
        rs_df = load_rs_from_cloud()

    ind_df = load_industry_from_cloud()

    # 사이드바 필터
    st.sidebar.header("필터")

    # 기본 범위
    min_price = st.sidebar.number_input("최소 주가(USD)", min_value=0.0, value=15.0, step=1.0)
    min_dollar_vol = st.sidebar.number_input(
        "최소 50일 평균 거래대금(USD)", min_value=0.0, value=5_000_000.0, step=1_000_000.0
    )
    min_rs = st.sidebar.slider("최소 RS (O'Neil 0~99)", min_value=0, max_value=99, value=80)

    st.sidebar.markdown("---")

    smr_grades_all = ["A", "B", "C", "D", "E"]
    selected_smr_grades = st.sidebar.multiselect(
        "SMR 등급 필터", smr_grades_all, default=["A", "B"]
    )

    st.sidebar.markdown("---")

    use_industry_filter = st.sidebar.checkbox("산업군 랭크/등급 필터 사용", value=True)
    max_group_rank = st.sidebar.number_input(
        "허용 최대 산업군 랭크 (작을수록 상위)", min_value=1, value=50, step=1
    )
    allowed_group_grades = st.sidebar.multiselect(
        "허용 산업군 등급", ["A", "B", "C", "D", "E"], default=["A", "B"]
    )

    st.sidebar.markdown("---")

    top_n = st.sidebar.number_input("표시할 최대 종목 수", min_value=10, max_value=500, value=100, step=10)

    # 필터 적용
    df = rs_df.copy()

    # 가격 / 거래대금 / RS 필터
    df = df[df["last_close"] >= min_price]
    df = df[df["avg_dollar_vol_50"] >= min_dollar_vol]
    df = df[df["rs_onil_99"] >= min_rs]

    # SMR 등급 필터
    df = df[df["smr_grade"].isin(selected_smr_grades)]

    # 산업군 필터 (선택)
    if use_industry_filter:
        df = df[df["group_rank"] <= max_group_rank]
        df = df[df["group_grade"].isin(allowed_group_grades)]

    # 정렬: RS 상위 → 산업군 RS 상위
    sort_cols = ["rs_onil_99", "group_rs_99"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    # 상위 N개만
    df = df.head(top_n)

    # 메인 레이아웃: 상단 요약 + 하단 탭
    st.subheader("요약")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("필터 후 종목 수", len(df))
    with col2:
        if "group_key" in df.columns:
            st.metric("산업군 수", df["group_key"].nunique())
    with col3:
        st.metric("RS 기준 하한", f"{min_rs}")
    with col4:
        st.metric("SMR 등급", ", ".join(selected_smr_grades))

    st.markdown("---")

    # 탭 구성: 랭킹 / 차트 / 재무
    tab_rank, tab_chart, tab_fund = st.tabs(["📊 랭킹 / 리스트", "🕹 차트(TradingView)", "📑 재무 / SMR"])

    # ------------------------------
    # 탭 1: 랭킹 / 리스트
    # ------------------------------
    with tab_rank:
        st.subheader("필터 적용 후 종목 리스트")

        if df.empty:
            st.warning("필터 조건에 해당하는 종목이 없습니다. 필터를 완화해 보세요.")
        else:
            # 표시용 테이블
            display_cols = [
                "ticker",
                "last_close",
                "rs_onil_99",
                "group_key",
                "group_rank",
                "group_grade",
                "group_rs_99",
                "ret_3m",
                "ret_6m",
                "ret_12m",
                "smr_grade",
                "sales_growth",
                "profit_margin",
                "roe",
                "avg_dollar_vol_50",
            ]
            display_cols = [c for c in display_cols if c in df.columns]

            disp = df[display_cols].copy()

            if "last_close" in disp.columns:
                disp["last_close"] = disp["last_close"].apply(format_price)
            if "ret_3m" in disp.columns:
                disp["ret_3m"] = disp["ret_3m"].apply(format_percentage)
            if "ret_6m" in disp.columns:
                disp["ret_6m"] = disp["ret_6m"].apply(format_percentage)
            if "ret_12m" in disp.columns:
                disp["ret_12m"] = disp["ret_12m"].apply(format_percentage)
            if "avg_dollar_vol_50" in disp.columns:
                disp["avg_dollar_vol_50"] = disp["avg_dollar_vol_50"].apply(short_k)
            if "sales_growth" in disp.columns:
                disp["sales_growth"] = disp["sales_growth"].apply(format_percentage)
            if "profit_margin" in disp.columns:
                disp["profit_margin"] = disp["profit_margin"].apply(format_percentage)
            if "roe" in disp.columns:
                disp["roe"] = disp["roe"].apply(format_percentage)

            st.dataframe(
                disp,
                use_container_width=True,
                height=400,
            )

    # ------------------------------
    # 공통: 종목 선택 위젯
    # ------------------------------
    st.markdown("---")

    if df.empty:
        st.info("차트/재무를 보기 위해서는 먼저 필터를 완화하여 종목이 나오도록 해야 합니다.")
        return

    tickers = df["ticker"].dropna().astype(str).unique().tolist()
    default_ticker = tickers[0] if tickers else None

    selected_ticker = st.selectbox(
        "상세 차트/재무를 볼 종목 선택",
        options=tickers,
        index=0 if default_ticker else None,
    )

    selected_row = df[df["ticker"] == selected_ticker].head(1)

    # ------------------------------
    # 탭 2: 차트 (TradingView)
    # ------------------------------
    with tab_chart:
        st.subheader(f"TradingView 차트 · {selected_ticker}")

        st.caption("※ TradingView에서 제공하는 웹 위젯으로 일봉 차트를 확인합니다.")
        render_tradingview_chart(selected_ticker)

    # ------------------------------
    # 탭 3: 재무 / SMR
    # ------------------------------
    with tab_fund:
        st.subheader(f"SMR 요약 · {selected_ticker}")

        if selected_row.empty:
            st.warning("선택한 종목 데이터를 찾을 수 없습니다.")
        else:
            row = selected_row.iloc[0]

            # 왼쪽: 기본 정보, 오른쪽: SMR 상세
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**기본 지표**")
                st.write(f"- Ticker: `{row['ticker']}`")
                if "last_close" in row:
                    st.write(f"- 종가: {format_price(row['last_close'])} USD")
                if "ret_3m" in row:
                    st.write(f"- 3M 수익률: {format_percentage(row['ret_3m'])}")
                if "ret_6m" in row:
                    st.write(f"- 6M 수익률: {format_percentage(row['ret_6m'])}")
                if "ret_12m" in row:
                    st.write(f"- 12M 수익률: {format_percentage(row['ret_12m'])}")
                if "rs_onil_99" in row:
                    st.write(f"- RS (0~99): {row['rs_onil_99']:.1f}")

                if "group_key" in row:
                    st.markdown("---")
                    st.markdown("**산업군 정보**")
                    st.write(f"- 그룹 키: {row['group_key']}")
                    if "group_rank" in row and not pd.isna(row["group_rank"]):
                        st.write(f"- 그룹 랭크: {int(row['group_rank'])}")
                    if "group_grade" in row:
                        st.write(f"- 그룹 등급: {row['group_grade']}")
                    if "group_rs_99" in row and not pd.isna(row["group_rs_99"]):
                        st.write(f"- 그룹 RS (0~99): {row['group_rs_99']:.1f}")

            with c2:
                st.markdown("**SMR 요약**")
                if "smr_grade" in row:
                    st.write(f"- SMR 등급: **{row['smr_grade']}**")
                if "smr_score" in row and not pd.isna(row["smr_score"]):
                    st.write(f"- SMR 점수 (0~100): {row['smr_score']:.1f}")

                st.markdown("---")

                if "sales_growth" in row:
                    st.write(f"- 매출 성장률(최근 연간 기준): {format_percentage(row['sales_growth'])}")
                if "profit_margin" in row:
                    st.write(f"- 이익률(최근 연간 기준): {format_percentage(row['profit_margin'])}")
                if "roe" in row:
                    st.write(f"- ROE(최근 연간 기준): {format_percentage(row['roe'])}")

                st.caption(
                    "※ SMR은 매출 성장(S), 이익률(M), ROE(R) 조합 점수로 계산한 내부 지표입니다."
                )


# ==============================
# 엔트리 포인트
# ==============================
if __name__ == "__main__":
    main()
