"""10-7 대체 시험: 보정 파일의 카메라 배치에 가상의 사람을 세워 ankle1 / stereo / fused 거리 계산을 비교한다
실제 카메라 영상 없이, 관절(어깨 중심·발목)을 두 카메라 화면에 투영하고 픽셀 잡음을 더한 뒤 proximity_app.py와
같은 식(stereo.py)으로 위치를 다시 구한다. 화면 밖이거나 가려진 관절은 "없음"이다.
사용법: python mode_sim.py --calib-dir calib/before_1103 --target 1.8,0 --noise 2
"""
import argparse
import numpy as np
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--calib-dir", default="calib")
ap.add_argument("--target", default="1.8,0")
ap.add_argument("--noise", type=float, default=2.0, help="관절 픽셀 잡음 표준편차(px)")
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--sizes", default="848x480,640x480", help="cam0,cam1 화면 크기")
ap.add_argument("--toward", action="store_true", help="사람을 타겟에서 카메라 쪽에 세움 (기본: 카메라 반대쪽, 타겟 뒤)")
a = ap.parse_args()

cams = [S.load_cam(f"{a.calib_dir}/cam{n}.npz") for n in (0, 1)]
sizes = [tuple(map(int, s.split("x"))) for s in a.sizes.split(",")]
T = np.array(list(map(float, a.target.split(","))))
H_GUESS, SHOULDER_STAND, SHOULDER_SIT, ANKLE = 1.2, 1.40, 1.00, 0.08
rng = np.random.default_rng(0)

def visible(n, X):
    c = cams[n]
    if (c["R"] @ np.asarray(X) + c["t"].ravel())[2] <= 0.1:     # 카메라 뒤
        return None
    u, v = S.project_distorted(c, X)
    w, h = sizes[n]
    return (u, v) if 0 <= u < w and 0 <= v < h else None

def noisy(p):
    return np.asarray(p) + rng.normal(0, a.noise, 2)

def dist_xy(X):
    return None if X is None else float(np.linalg.norm(np.asarray(X)[:2] - T))

def run(xy, shoulder_h, ankle_hidden):
    S3, A3 = [*xy, shoulder_h], [*xy, ANKLE]
    vs = [visible(n, S3) for n in (0, 1)]
    va = None if ankle_hidden else visible(1, A3)                 # ankle1은 C920(cam1) 단독 (보고서 10-3과 같음)
    out = {"ankle1": [], "stereo": [], "fused": []}
    for _ in range(a.trials):
        if va is not None:
            X = S.ray_plane(cams[1], S.undistort(cams[1], [noisy(va)])[0], 0.0)
            out["ankle1"].append(dist_xy(X))
        if vs[0] is not None and vs[1] is not None:
            p = [S.undistort(cams[n], [noisy(vs[n])])[0] for n in (0, 1)]
            X, _ = S.triangulate(cams[0], cams[1], p[0], p[1])
            out["stereo"].append(dist_xy(X)); out["fused"].append(dist_xy(X))
        elif vs[0] is not None or vs[1] is not None:
            n = 0 if vs[0] is not None else 1
            X = S.ray_plane(cams[n], S.undistort(cams[n], [noisy(vs[n])])[0], H_GUESS)
            out["fused"].append(dist_xy(X))
    fmt = lambda v: "없음" if not v or any(x is None for x in v) else f"{np.median(v):.2f} (±{np.std(v):.2f})"
    return {k: fmt(v) for k, v in out.items()}, vs

C = [c["C"] for c in cams]
mid = (C[0][:2] + C[1][:2]) / 2
u = (mid - T) / np.linalg.norm(mid - T)                          # 타겟에서 카메라 쪽 방향
if not a.toward:
    u = -u                                                       # 카메라에 너무 가까우면 몸이 화면 밖으로 나가므로 타겟 뒤쪽에 세운다
side = np.array([-u[1], u[0]])
print(f"방향 {'카메라 쪽' if a.toward else '타겟 뒤쪽'}, 카메라 위치 cam0 {np.round(C[0], 2)}, cam1 {np.round(C[1], 2)}, 타겟 {T}, 잡음 {a.noise}px, {a.trials}회")
cases = [("0.5 m, 서 있음", 0.5, SHOULDER_STAND, False), ("1.0 m, 서 있음", 1.0, SHOULDER_STAND, False),
         ("2.0 m, 서 있음", 2.0, SHOULDER_STAND, False), ("3.0 m, 서 있음", 3.0, SHOULDER_STAND, False),
         ("1.0 m, 책상에 앉음", 1.0, SHOULDER_SIT, True), ("2.0 m, 책상 뒤에 서 있음", 2.0, SHOULDER_STAND, True)]
print("| 실제 거리·자세 | ankle1 | stereo | fused |\n| --- | --- | --- | --- |")
for name, d, sh, hid in cases:
    r, _ = run(T + u * d, sh, hid)
    print(f"| {name} | {r['ankle1']} | {r['stereo']} | {r['fused']} |")
for off in np.arange(0.5, 4.01, 0.25):                           # 한 카메라에만 보이는 옆 위치 찾기
    for sgn in (1, -1):
        xy = T + u * 2.0 + side * off * sgn
        r, vs = run(xy, SHOULDER_STAND, False)
        if (vs[0] is None) != (vs[1] is None):
            real = float(np.linalg.norm(xy - T))
            print(f"| 한 카메라에만 보임 (cam{0 if vs[0] is not None else 1}, 실제 {real:.2f} m) | {r['ankle1']} | {r['stereo']} | {r['fused']} |")
            raise SystemExit
print("| 한 카메라에만 보이는 위치 | 찾지 못함 | | |")
