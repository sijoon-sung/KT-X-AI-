"""손으로 계산 가능한 합성 이력으로 재고/노출 복원을 검증.

자전거 A: ST1 08:00 대여 → ST2 08:30 반납, ST2 10:15 대여 → ST1 10:45 반납
자전거 B: ST2 09:00 대여 → ST2 09:10 반납, ST1 12:00 대여 → ST1 12:10 반납 (ST2→ST1 재배치)
"""
import csv, io, subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
HDR = ["자전거번호", "대여일시", "대여_대여소ID", "대여_대여소명", "대여_X좌표", "대여_Y좌표", "대여_구", "대여_동", "대여_대여소주소",
       "반납일시", "반납_대여소ID", "반납_대여소명", "반납_X좌표", "반납_Y좌표", "반납_구", "반납_동", "반납_대여소주소", "이용시간(분)", "이용거리(km)"]
ST = {"ST1": ("가", "36.35", "127.38"), "ST2": ("나", "36.36", "127.39"), "ST3": ("다", "36.37", "127.40")}
TRIPS = [("A", "2024-01-01 08:00:00", "ST1", "2024-01-01 08:30:00", "ST2"),
         ("B", "2024-01-01 09:00:00", "ST2", "2024-01-01 09:10:00", "ST2"),
         ("A", "2024-01-01 10:15:00", "ST2", "2024-01-01 10:45:00", "ST1"),
         ("B", "2024-01-01 12:00:00", "ST1", "2024-01-01 12:10:00", "ST1"),
         # C: ST3 에서 13:20 반납 후 1분 뒤 같은 자전거 재대여 → 한 이용(13:00 대여, 13:40 반납)으로 합쳐져야 함
         ("C", "2024-01-01 13:00:00", "ST3", "2024-01-01 13:20:00", "ST3"),
         ("C", "2024-01-01 13:21:00", "ST3", "2024-01-01 13:40:00", "ST3")]


def row(b, rt, rs, tt, ts):
    return [b, rt, rs, *ST[rs], "구", "동", "주소", tt, ts, *ST[ts], "구", "동", "주소", "10", "1.0"]


def test_reconstruction():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        buf = io.StringIO(); w = csv.writer(buf); w.writerow(HDR)
        for t in TRIPS:
            w.writerow(row(*t))
        with zipfile.ZipFile(tmp / "t.zip", "w") as zf:
            zf.writestr("t.csv", buf.getvalue().encode("cp949"))
        subprocess.run([sys.executable, str(HERE / "preprocess.py"), "--zip", str(tmp / "t.zip"), "--out", str(tmp),
                        "--min-trips-per-day", "0"], check=True, capture_output=True)
        with np.load(tmp / "tashu_60min.npz") as f:
            z = {k: f[k] for k in f.files}
        s1, s2 = list(z["stations"]).index("ST1"), list(z["stations"]).index("ST2")
        inv, ex, r = z["inv_start"], z["exposure"], z["rentals"]

        assert r[8, s1] == 1 and r[9, s2] == 1 and r[10, s2] == 1 and r[12, s1] == 1
        # ST2: B 가 00:00~09:00, A 가 08:30~10:15 정차
        assert inv[0, s2] == 1 and inv[8, s2] == 1 and inv[9, s2] == 1 and inv[10, s2] == 1 and inv[11, s2] == 0
        assert np.allclose(ex[:10, s2], 1) and np.isclose(ex[10, s2], 0.25) and ex[11, s2] == 0
        # ST1: A 00:00~08:00, 10:45~ / B 는 재배치 도착(시각 미상) → 대여 직전 1초만, 12:10~
        assert inv[7, s1] == 1 and inv[8, s1] == 0 and inv[11, s1] == 1 and inv[12, s1] == 1 and inv[13, s1] == 2
        assert ex[8, s1] == 0 and ex[9, s1] == 0 and np.isclose(ex[10, s1], 0.25) and ex[11, s1] == 1
        assert z["censored"][10, s2] and not z["censored"][9, s2]
        # 상한: 재배치된 B 가 09:10~12:00 동안 ST2 와 ST1 양쪽에 있었다고 봄
        assert z["inv_ub"][10, s2] == 2 and z["inv_ub"][11, s2] == 1 and z["inv_ub"][12, s2] == 0
        assert np.isclose(z["exposure_ub"][9, s1], 50 / 60) and z["exposure_ub"][11, s2] == 1
        s3 = list(z["stations"]).index("ST3")
        assert r[13, s3] == 1 and z["returns"][13, s3] == 1
        assert np.isclose(ex[13, s3], 20 / 60)                    # 00:00~13:00 정차 후 13:40 반납부터 다시
        rel = (tmp / "relocations.csv").read_text(encoding="utf-8-sig").splitlines()
        assert len(rel) == 2 and rel[1].startswith("ST2,ST1")
    print("ok")


if __name__ == "__main__":
    test_reconstruction()
