# clean_old_files.py
#
# 역할:
# - rs_onil_all_YYYYMMDD*.csv
# - industry_rs_6m_YYYYMMDD*.csv
# 처럼 날짜가 박힌 요약 파일들 중,
# "최근 N일"만 남기고 나머지를 자동 삭제한다.

import os
import glob
import datetime as dt

# 최근 몇 일치 데이터만 남길지 (원하는 대로 바꿔도 됨)
KEEP_DAYS = 7  # 최근 7일치만 유지

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cleanup(pattern: str, prefix: str):
    """
    pattern: glob 패턴 (예: 'rs_onil_all_*.csv')
    prefix : 파일명 앞부분 (예: 'rs_onil_all_')
    파일명에서 YYYYMMDD를 뽑아 날짜 기준으로 KEEP_DAYS 이전 것은 삭제.
    """
    path_pattern = os.path.join(BASE_DIR, pattern)
    files = glob.glob(path_pattern)

    if not files:
        print(f"[INFO] 패턴 {pattern} 에 해당하는 파일이 없습니다.")
        return

    today = dt.date.today()
    # 예: KEEP_DAYS=7이면, 오늘~6일 전까지는 유지, 그 이전은 삭제
    keep_threshold = today - dt.timedelta(days=KEEP_DAYS - 1)

    print(f"[INFO] 패턴 {pattern} 정리 시작 (최근 {KEEP_DAYS}일만 유지, 기준일: {keep_threshold} 이후)")

    deleted = 0
    for f in files:
        base = os.path.basename(f)

        # prefix 이후 문자열에서 앞 8글자를 날짜(YYYYMMDD)로 가정
        # 예: rs_onil_all_20251207_smr.csv
        #  -> 남는 부분 = '20251207_smr.csv'
        #  -> date_str = '20251207'
        rest = base.replace(prefix, "", 1)
        rest = rest.split(".")[0]  # 확장자 제거 -> '20251207_smr'
        date_str = rest[:8]

        try:
            file_date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            print(f"[SKIP] 날짜 파싱 실패, 삭제 대상 아님: {base}")
            continue

        if file_date < keep_threshold:
            try:
                os.remove(f)
                print(f"[DELETE] {base}")
                deleted += 1
            except Exception as e:
                print(f"[ERROR] 삭제 실패: {base} ({e})")

    print(f"[INFO] 패턴 {pattern} 정리 완료, 삭제 개수: {deleted}")


def main():
    print(f"=== 오래된 요약 파일 자동 정리 시작 (최근 {KEEP_DAYS}일만 유지) ===")

    # 1) 개별 RS + SMR 요약 파일 (rs_onil_all_YYYYMMDD*.csv)
    cleanup("rs_onil_all_*.csv", "rs_onil_all_")

    # 2) 산업군 RS 요약 파일 (industry_rs_6m_YYYYMMDD*.csv)
    cleanup("industry_rs_6m_*.csv", "industry_rs_6m_")

    print("=== 정리 완료 ===")


if __name__ == "__main__":
    main()
