"""10-2 카메라 위치 보정 (체커보드 방식): 두 카메라에 동시에 보이는 체커보드 하나로 위치·방향을 구한다
줄자로 기준점을 재는 calib_extrinsics.py 대신 쓴다. 결과 형식(calib/cam0.npz, cam1.npz)은 같다.
좌표계: 원점 = 체커보드 중심 바로 아래 바닥, x = 카메라가 보는 방향(앞 +), y = 왼쪽(+), z = 위(+), 단위 m
사용법: python calib_board.py --cams vigi 0 --square 0.024 --pose upright --z 1.00   (세워 둠, z = 체커보드 중심 높이)
        python calib_board.py --cams vigi 0 --square 0.024 --pose flat --z 0.72      (눕혀 둠, z = 체커보드 면 높이)
"""
import argparse, json, os, threading, time
import cv2, numpy as np
import stereo as S
from camsrc import open_capture, cam_name

def find_corners(frame, size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, c = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit).reshape(-1, 2)
    found, c = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_EXHAUSTIVE)
    return c.reshape(-1, 2) if found else None

class Latest(threading.Thread):
    """카메라의 가장 최근 프레임만 보관"""
    def __init__(self, spec, res):
        super().__init__(daemon=True)
        self.cap = open_capture(spec, *res)
        assert self.cap is not None, f"카메라 {spec}을 열 수 없음"
        self.frame = None
    def run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

def grab_all(specs, size, res, n=8, moved=3.0, timeout=60.0):
    """모든 카메라에서 체커보드가 움직이지 않는 프레임 n묶음의 코너 평균과 마지막 프레임 (손으로 들어도 멈추면 된다)
    n묶음 안에서 코너가 moved(px)보다 움직이면 처음부터 다시 모은다. 못 모으면 찾은 카메라만 코너를 돌려준다"""
    srcs = [Latest(s, res) for s in specs]
    for s in srcs:
        s.start()
    time.sleep(3.0)                 # 자동 노출 안정화
    t0, got, last, frames = time.time(), [], [None] * len(srcs), [None] * len(srcs)
    while time.time() - t0 < timeout and len(got) < n:
        frames = [s.frame for s in srcs]
        if any(f is None for f in frames):
            time.sleep(0.05); continue
        cs = [find_corners(f, size) for f in frames]
        for i, c in enumerate(cs):
            if c is not None:
                if got and np.linalg.norm(c[0] - got[0][i][0]) > np.linalg.norm(c[-1] - got[0][i][0]):
                    c = c[::-1]     # 같은 코너가 먼저 오도록 순서 통일
                    cs[i] = c
                last[i] = c
        if any(c is None for c in cs):
            got = []; continue
        if got and max(np.linalg.norm(c - g, axis=1).mean() for c, g in zip(cs, got[0])) > moved:
            got = []
        got.append(cs)
        if len(got) in (1, n):
            print(f"  체커보드 멈춤 확인 {len(got)}/{n}")
    corners = [np.mean([g[i] for g in got], axis=0) for i in range(len(srcs))] if len(got) >= n else last
    return corners, frames, len(got) >= n

def board_points(cols, rows, square):
    """체커보드 좌표: 중심이 원점, x = 가로, y = 세로(위 +), z = 앞면 쪽(+)"""
    p = np.zeros((rows * cols, 3), np.float64)
    p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    p[:, 1] *= -1
    return p - p.mean(0)

def pose(obj, img, K, dist):
    """평면 물체의 해 2개 중 카메라가 앞면 쪽에 있고 재투영 오차가 작은 것"""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for r, t, e in zip(rvecs, tvecs, np.ravel(errs)):
        R, _ = cv2.Rodrigues(r)
        C = (-R.T @ t).ravel()
        if C[2] > 0 and (best is None or e < best[2]):
            best = (R, t.reshape(3, 1), float(e), C)
    return best

def solve(obj, imgs, Ks, dists, flat, z):
    """카메라별 위치·방향을 방 좌표로 변환. 반환: [(R, t, 재투영 px)]"""
    p0 = pose(obj, imgs[0], Ks[0], dists[0])
    poses = [p0]
    if len(imgs) == 2:             # 체커보드는 180도 돌려도 같아 보인다 → 두 카메라의 방향이 더 비슷한 쪽이 올바른 순서
        orders = [imgs[1], imgs[1][::-1]]
        cands = [pose(obj, im, Ks[1], dists[1]) for im in orders]
        i = max((0, 1), key=lambda k: np.trace(p0[0] @ cands[k][0].T))
        poses.append(cands[i])
        imgs[1] = orders[i]        # 삼각측량 확인에도 같은 순서를 쓴다
    up = np.array([0.0, 0.0, 1.0]) if flat else np.array([0.0, 1.0, 0.0])
    if up @ (p0[0].T @ np.array([0.0, -1.0, 0.0])) < 0:          # 코너 순서가 뒤집혀 찾혔으면 위쪽 방향도 뒤집는다
        up = -up
    f = sum(p[0].T @ np.array([0.0, 0.0, 1.0]) for p in poses)   # 카메라들이 보는 방향의 평균
    f -= (f @ up) * up
    f /= np.linalg.norm(f)
    Rwb = np.stack([f, np.cross(up, f), up])      # 체커보드 → 방 좌표 회전
    T = np.array([[0.0], [0.0], [z]])
    out = []
    for R, t, e, _ in poses:
        Rn = R @ Rwb.T
        out.append((Rn, t - Rn @ T, e))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["vigi", "0"], help="calib_intrinsics.py의 --cam 값을 순서대로 (1~2개)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, required=True, help="칸 한 변 길이(m)")
    ap.add_argument("--pose", choices=["upright", "flat"], default="upright", help="체커보드를 세워 둠 / 눕혀 둠")
    ap.add_argument("--z", type=float, required=True, help="upright: 체커보드 중심 높이, flat: 체커보드 면 높이 (m)")
    ap.add_argument("--room", type=float, nargs=4, default=[-2, -6, 8, 6], metavar=("X0", "Y0", "X1", "Y1"),
                    help="사람을 찾을 바닥 범위(m). proximity_app.py가 씀")
    ap.add_argument("--res", default="640x480", help="USB 웹캠 요청 해상도")
    args = ap.parse_args()
    res = tuple(map(int, args.res.split("x")))
    size = (args.cols, args.rows)
    os.makedirs("calib", exist_ok=True)

    print("체커보드를 두 화면에 보이게 하고 움직이지 마세요. 최대 60초 기다립니다")
    imgs, frames, ok = grab_all(args.cams, size, res)
    shapes, Ks, dists = [], [], []
    for n, spec in enumerate(args.cams):
        c, f = imgs[n], frames[n]
        if f is not None:
            vis = f.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, size, c.astype(np.float32).reshape(-1, 1, 2), True)
            cv2.imwrite(f"calib/board{n}.jpg", vis)
        print(f"cam{n} ({spec}): {'체커보드 찾음' if c is not None else '체커보드 못 찾음'}  → calib/board{n}.jpg")
        d = np.load(f"calib/{cam_name(spec)}_intr.npz")
        shapes.append(f.shape[1::-1] if f is not None else None); Ks.append(d["K"]); dists.append(d["dist"])
    if not ok:
        raise SystemExit("체커보드가 두 화면에 함께, 멈춘 상태로 잡히지 않음. calib/board*.jpg 를 보고 위치를 조정한 뒤 다시 실행")

    obj = board_points(args.cols, args.rows, args.square)
    res_poses = solve(obj, imgs, Ks, dists, args.pose == "flat", args.z)
    for n, (R, t, e) in enumerate(res_poses):
        np.savez(f"calib/cam{n}.npz", K=Ks[n], dist=dists[n], R=R, t=t, res=np.array(shapes[n]))
        print(f"cam{n} 위치 (x, y, z) = {(-R.T @ t).ravel().round(3).tolist()} m, 재투영 {e:.2f}px")

    if len(res_poses) == 2:
        c0, c1 = S.load_cam("calib/cam0.npz"), S.load_cam("calib/cam1.npz")
        print(f"두 카메라 간격 {np.linalg.norm(c0['C'] - c1['C']):.3f} m  ← 줄자로 잰 값과 비교")
        p0, p1 = S.undistort(c0, imgs[0]), S.undistort(c1, imgs[1])
        X = np.array([S.triangulate(c0, c1, a, b)[0] for a, b in zip(p0, p1)])
        step = np.linalg.norm(np.diff(X[:args.cols], axis=0), axis=1).mean()
        print(f"삼각측량한 칸 크기 {step * 1000:.1f} mm (실제 {args.square * 1000:.1f} mm)")

    x0, y0, x1, y1 = args.room
    pts = [{"id": f"R{i}", "xyz": [x, y, 0.0]} for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]
    if os.path.exists("calib/world_points.json"):
        os.replace("calib/world_points.json", "calib/world_points_bak.json")
    json.dump({"_설명": "calib_board.py가 만든 바닥 범위. pad만큼 더해 사람을 찾는다", "points": pts},
              open("calib/world_points.json", "w"), ensure_ascii=False, indent=1)
    print("저장: calib/cam*.npz, calib/world_points.json")

if __name__ == "__main__":
    main()
