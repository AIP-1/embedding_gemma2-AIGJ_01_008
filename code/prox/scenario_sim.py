"""10-6·10-7·10-8 가상 시험: 실제 보정 파일의 카메라 배치에 가상의 사람과 카트를 움직여, 앱과 같은 함수로 알람을 낸다

- 카메라: calib/cam0.npz(VIGI, 848x480), cam1.npz(C920, 640x480)의 렌즈·위치를 그대로 쓴다.
- 관절과 카트 중심을 두 화면에 투영하고 픽셀 잡음(기본 2px)을 더해 stereo.triangulate로 위치를 다시 구한다.
- 알람은 proximity_app.py의 ProximityAlarm 클래스 원문을 그대로 불러 쓴다(앱 실행 없이).
- 사람 추적은 1명으로 단순화했다: 0.5초 중앙값(앱의 SMOOTH_S)으로 위치·높이를 정한다.
사용법: python scenario_sim.py --calib-dir calib/before_1103 [--part 106|108|layout]
"""
import argparse, re, types
import numpy as np
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--calib-dir", default="calib")
ap.add_argument("--target", default="1.8,0")
ap.add_argument("--noise", type=float, default=2.0)
ap.add_argument("--fps", type=float, default=20.0)
ap.add_argument("--part", default="all", choices=["all", "106", "108", "layout"])
a = ap.parse_args()

# ---- 앱의 알람 클래스를 원문 그대로 가져온다 ----
src = open("proximity_app.py", encoding="utf-8").read()
cls = src[src.index("class ProximityAlarm"):src.index("# ---------------- 바닥 평면 추적")]
ns = {"args": types.SimpleNamespace(ttc=2.0), "CLOSE_V": 0.5}
exec(cls, ns)
ProximityAlarm = ns["ProximityAlarm"]
SIT_H, SMOOTH_S, MAX_ERR = 1.15, 0.5, 15.0

REAL = [S.load_cam(f"{a.calib_dir}/cam{n}.npz") for n in (0, 1)]
SIZES = [(848, 480), (640, 480)]
T = np.array(list(map(float, a.target.split(","))))
mid = (REAL[0]["C"][:2] + REAL[1]["C"][:2]) / 2
U = (T - mid) / np.linalg.norm(T - mid)          # 타겟에서 카메라 반대쪽 방향
SIDE = np.array([-U[1], U[0]])
DT = 1.0 / a.fps

def visible(cam, size, X):
    if (cam["R"] @ np.asarray(X, float) + cam["t"].ravel())[2] <= 0.1:
        return None
    u, v = S.project_distorted(cam, X)
    return (u, v) if 0 <= u < size[0] and 0 <= v < size[1] else None

def measure(cams, X, rng, hidden=()):
    """두 카메라로 삼각측량. 한쪽만 보이면 (None, 그 카메라 번호, 픽셀)"""
    pts = [None if n in hidden else visible(cams[n], SIZES[n], X) for n in (0, 1)]
    noisy = [None if p is None else S.undistort(cams[n], [np.asarray(p) + rng.normal(0, a.noise, 2)])[0]
             for n, p in enumerate(pts)]
    if noisy[0] is not None and noisy[1] is not None:
        Xe, err = S.triangulate(cams[0], cams[1], noisy[0], noisy[1])
        return (Xe, None, None) if err <= MAX_ERR else (None, None, None)
    for n in (0, 1):
        if noisy[n] is not None:
            return None, n, noisy[n]
    return None, None, None

# ---------------- 10-6: 사람 시나리오 ----------------
def run_person(path, rng, near=1.0, far=3.0):
    """path: [(t, xy, 어깨 높이)]. 반환: 이벤트 목록, 마지막 자세"""
    al, hist, ev, pose = ProximityAlarm(near, far), [], [], "-"
    for t, xy, h in path:
        X, _, _ = measure(REAL, [*xy, h], rng)
        if X is None:
            continue
        hist = [(tt, p) for tt, p in hist if t - tt < SMOOTH_S] + [(t, X)]
        P = np.median(np.array([p for _, p in hist]), axis=0)
        pose = "앉음" if P[2] < SIT_H else "섬"
        ev += [(round(t, 1), e) for e in al.update(t, float(np.linalg.norm(P[:2] - T)))]
    return ev, pose

def walk(p0, p1, v, h, t0=0.0):
    d = np.linalg.norm(np.subtract(p1, p0)); n = max(int(d / v / DT), 1)
    return [(t0 + i * DT, np.asarray(p0) + (np.asarray(p1) - p0) * i / n, h) for i in range(n + 1)]

def stand(p, secs, h, t0):
    return [(t0 + i * DT, np.asarray(p), h) for i in range(int(secs / DT) + 1)]

def part106():
    sc = {}
    def approach():
        w = walk(T + U * 4, T + U * 0.5, 1.0, 1.40); return w + stand(T + U * 0.5, 2, 1.40, w[-1][0] + DT)
    def leave():
        s = stand(T + U * 0.5, 1, 1.40, 0); return s + walk(T + U * 0.5, T + U * 4, 1.0, 1.40, s[-1][0] + DT)
    def boundary():
        return stand(T + U * 1.0, 10, 1.40, 0)
    def passby():
        return walk(T + U * 2 - SIDE * 2.5, T + U * 2 + SIDE * 2.5, 1.0, 1.40)
    def sit():
        w = walk(T + U * 3, T + U * 0.8, 1.0, 1.40); t = w[-1][0]
        down = [(t + i * DT, T + U * 0.8, 1.40 - 0.40 * i / int(1 / DT)) for i in range(1, int(1 / DT) + 1)]
        return w + down + stand(T + U * 0.8, 5, 1.00, down[-1][0] + DT)
    rows = [("4m에서 타겟으로 걸어옴", "접근 시작 → 알람: 근접", approach,
             lambda ev, po: any("근접" in e for _, e in ev)),
            ("타겟 앞 0.5m에서 4m 밖으로 나감", "알람: 이탈", leave, lambda ev, po: any("이탈" in e for _, e in ev)),
            ("1.0m 경계에서 10초 서 있음", "알람 1회만", boundary, lambda ev, po: sum("근접" in e for _, e in ev) == 1),
            ("타겟 2m 옆을 지나감", "근접 알람 없음", passby, lambda ev, po: not any("근접" in e for _, e in ev)),
            ("타겟 옆 책상에 앉음 (0.8m)", "근접 알람, 자세 = 앉음", sit,
             lambda ev, po: any("근접" in e for _, e in ev) and po == "앉음")]
    print("\n### 10-6 가상 시나리오 (stereo, 근접 1.0m / 이탈 3.0m, 3회)")
    print("| 시나리오 (3회씩) | 기대 결과 | 맞은 횟수 | 나온 이벤트 (1회차) |\n| --- | --- | --- | --- |")
    for name, want, make, ok in rows:
        res = [run_person(make(), np.random.default_rng(s)) for s in range(3)]
        hit = sum(ok(ev, po) for ev, po in res)
        first = ", ".join(e for _, e in res[0][0]) or "없음"
        print(f"| {name} | {want} | {hit}/3 | {first}; 자세 {res[0][1]} |")

# ---------------- 10-8: 이동 타겟 ----------------
def run_cart(v, rng, hidden_win=None, near=1.0, far=3.0, stop_at=0.5, cart_h=0.5):
    person = T.copy(); start = T + U * 4
    kf_t = kf_p = None; al = ProximityAlarm(near, far); hist = []
    t, t_danger = 0.0, None; d_danger = v_est = t_near = None; t_stop = None; under = 0.0; n = 0
    pos = start.copy()
    while t < 12:
        d_true = float(np.linalg.norm(pos - person))
        if d_true > stop_at:
            pos = pos - U * v * DT
        elif t_stop is None:
            t_stop = t
        if t_stop is not None and t - t_stop > 2:
            break
        # 사람: 매 프레임 삼각측량 + 0.5초 중앙값
        X, _, _ = measure(REAL, [*person, 1.40], rng)
        if X is not None:
            hist = [(tt, p) for tt, p in hist if t - tt < SMOOTH_S] + [(t, X[:2])]
            pxy = np.median(np.array([p for _, p in hist]), axis=0)
            if kf_p is None: kf_p = S.Kalman2D(pxy)
            else: kf_p.predict(DT); kf_p.update(pxy)
        # 카트: 2프레임마다 검출 (--target-every 2), 가려지는 구간은 한 카메라만
        if kf_t is not None: kf_t.predict(DT)
        if n % 2 == 0:
            hid = (1,) if hidden_win and hidden_win[0] <= t <= hidden_win[1] else ()
            Xc, cam, pix = measure(REAL, [*pos, cart_h], rng, hidden=hid)
            if Xc is None and cam is not None:
                Xr = S.ray_plane(REAL[cam], pix, cart_h); Xc = Xr
            if Xc is not None:
                if kf_t is None: kf_t = S.Kalman2D(Xc[:2])
                else: kf_t.update(Xc[:2])
        n += 1
        if kf_t is not None and kf_p is not None:
            d = float(np.linalg.norm(kf_p.x[:2] - kf_t.x[:2]))
            for e in al.update(t, d):
                if "근접" in e and t_near is None: t_near = t
            dd, closing, ttc = S.relative_state(kf_p.x, kf_t.x)
            for e in al.update_ttc(closing, ttc, dd):
                if t_danger is None:
                    t_danger, d_danger, v_est = t, float(np.linalg.norm(pos - person)), float(np.linalg.norm(kf_t.x[2:]))
            if t_stop is not None:
                under = max(under, float(np.linalg.norm(pos - person)) - dd)
        t += DT
    lead = None if (t_danger is None or t_near is None) else t_near - t_danger
    return d_danger, lead, v_est, under

def part108():
    print(f"\n### 10-8 가상 이동 타겟 (사람 제자리, 카트 4m에서 출발, 0.5m에서 멈춤, {a.fps:.0f}fps, 3회 평균)")
    print("| 카트 속도 | 충돌 위험 알람 때 거리 (m) | 근접 알람보다 먼저 울린 시간 (초) | 추정 속도 (m/s) | 멈춘 뒤 거리 과소 추정 (m) |")
    print("| --- | --- | --- | --- | --- |")
    for name, v, hid in [("천천히 (0.5m/s)", 0.5, None), ("보통 (1.0m/s)", 1.0, None), ("빠르게 (1.5m/s)", 1.5, None),
                         ("C920에서 2.5초 가려짐 (1.0m/s)", 1.0, (0.5, 3.0))]:
        rs = [run_cart(v, np.random.default_rng(s), hid) for s in range(3)]
        f = lambda i: "알람 없음" if any(r[i] is None for r in rs) else f"{np.mean([r[i] for r in rs]):.2f}"
        print(f"| {name} | {f(0)} | {f(1)} | {f(2)} | {f(3)} |")

# ---------------- 10-7 추가: 카메라 배치 비교 ----------------
def look_cam(base, C, at):
    f = np.asarray(at, float) - C; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); d = np.cross(f, r)
    R = np.vstack([r, d, f]); t = (-R @ C).reshape(3, 1)
    return {"K": base["K"], "dist": base["dist"], "R": R, "t": t, "P": base["K"] @ np.hstack([R, t]),
            "C": np.asarray(C, float), "Kinv": base["Kinv"]}

def part_layout():
    Q = T + U * 1.5
    def pair(c0, c1, at_h=1.2):
        at = [*Q, at_h]
        return [look_cam(REAL[0], np.asarray(c0, float), at), look_cam(REAL[1], np.asarray(c1, float), at)]
    back = mid - U * 0.0
    layouts = [("지금 배치 (나란히, 높이 0.8~1.0m, 앞을 봄)", REAL),
               ("간격 1.5m, 높이 1.0m, 가운데를 봄", pair([*(back + SIDE * 0.75), 1.0], [*(back - SIDE * 0.75), 1.0])),
               ("간격 3m, 높이 2.4m, 가운데를 봄", pair([*(back + SIDE * 1.5), 2.4], [*(back - SIDE * 1.5), 2.4])),
               ("두 모서리, 시선 약 90°, 높이 2.4m", pair([*(Q - U * 3 + SIDE * 3), 2.4], [*(Q - U * 3 - SIDE * 3), 2.4]))]
    print(f"\n### 10-7 추가: 카메라 배치에 따른 stereo 위치 오차 (서 있는 사람, 잡음 {a.noise}px, 200회 평균, 단위 cm)")
    print("| 배치 | 두 카메라 간격 | 0.5m | 1.0m | 2.0m | 3.0m |\n| --- | --- | --- | --- | --- | --- |")
    for name, cams in layouts:
        gap = np.linalg.norm(cams[0]["C"] - cams[1]["C"])
        cells = []
        for dist in (0.5, 1.0, 2.0, 3.0):
            truth = T + U * dist; rng = np.random.default_rng(0); errs = []
            for _ in range(200):
                X, _, _ = measure(cams, [*truth, 1.40], rng)
                if X is not None: errs.append(np.linalg.norm(X[:2] - truth))
            cells.append(f"{100 * np.mean(errs):.0f}" if len(errs) > 100 else "안 보임")
        print(f"| {name} | {gap:.2f}m | " + " | ".join(cells) + " |")

print(f"카메라 cam0 {np.round(REAL[0]['C'], 2)}, cam1 {np.round(REAL[1]['C'], 2)}, 타겟 {T}, 잡음 {a.noise}px, {a.fps:.0f}fps")
if a.part in ("all", "106"): part106()
if a.part in ("all", "108"): part108()
if a.part in ("all", "layout"): part_layout()
