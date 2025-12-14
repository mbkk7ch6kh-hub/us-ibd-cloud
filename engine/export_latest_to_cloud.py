import os
import glob
import shutil

# 현재 파일 기준 디렉터리
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ENGINE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)


def find_latest(pattern: str, required: bool = True) -> str | None:
    """
    engine 디렉터리에서 pattern에 맞는 파일 중
    가장 최근(파일명 기준으로 정렬) 파일을 찾는다.
    """
    search_pattern = os.path.join(ENGINE_DIR, pattern)
    files = glob.glob(search_pattern)

    if not files:
        if required:
            raise FileNotFoundError(f"패턴에 맞는 파일을 찾지 못했습니다: {pattern}")
        return None

    # 파일명 기준 내림차순 정렬 (날짜가 들어있으므로 최신이 앞쪽)
    files.sort(reverse=True)
    return files[0]


def main():
    print("=== export_latest_to_cloud.py 시작 ===")
    print(f"[INFO] ENGINE_DIR = {ENGINE_DIR}")
    print(f"[INFO] DATA_DIR   = {DATA_DIR}")

    # 1) RS + SMR 파일: rs_onil_all_YYYYMMDD_smr.csv 가 있으면 그걸 우선 사용
    rs_path = find_latest("rs_onil_all_*_smr.csv", required=False)
    if rs_path is None:
        # SMR까지 붙은 파일이 없으면, RS 원본만이라도 사용
        print("[WARN] *_smr.csv 파일이 없어, rs_onil_all_*.csv 중 최신 파일을 사용합니다.")
        rs_path = find_latest("rs_onil_all_*.csv", required=True)

    # 2) 산업군 RS 파일
    ind_path = find_latest("industry_rs_6m_*.csv", required=True)

    latest_rs = os.path.join(DATA_DIR, "latest_rs_smr.csv")
    latest_ind = os.path.join(DATA_DIR, "latest_industry_rs.csv")

    shutil.copy2(rs_path, latest_rs)
    shutil.copy2(ind_path, latest_ind)

    print(f"[INFO] 최신 RS 파일 복사 완료: {rs_path} -> {latest_rs}")
    print(f"[INFO] 최신 산업군 RS 파일 복사 완료: {ind_path} -> {latest_ind}")
    print("=== export_latest_to_cloud.py 종료 ===")


if __name__ == "__main__":
    main()
