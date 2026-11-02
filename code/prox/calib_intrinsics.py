"""10-1 렌즈 보정: 체커보드를 여러 각도로 비추면 자동으로 20장을 모아 K(내부 행렬), dist(왜곡)를 저장
화면이 없어도 동작한다 (진행 상황은 터미널, 마지막 검출 이미지는 calib/<이름>_last.jpg)
사용법: python calib_intrinsics.py --cam vigi          (VIGI 카메라)
        python calib_intrinsics.py --cam 0             (USB 웹캠 /dev/video0)
체커보드를 모니터·태블릿 화면에 띄울 때는 화면의 칸 한 변을 자로 재서 --square 에 m 단위로 넣는다.
"""
import argparse, os, time
import cv2, numpy as np
from camsrc import open_capture, cam_name

ap = argparse.ArgumentParser()
ap.add_argument("--cam", required=True, help="vigi, rtsp://..., 또는 USB 장치 번호")
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480", help="USB 웹캠 요청 해상도 (RTSP는 스트림 해상도 그대로)")
ap.add_argument("--fix-center", action="store_true", help="렌즈 중심을 화면 중심, fx=fy로 고정 (왜곡이 작은 웹캠에서 cx·cy가 튈 때)")
ap.add_argument("--resolve", action="store_true", help="다시 촬영하지 않고 저장된 코너(calib/<이름>_pts.npz)로 다시 계산")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
NAME = cam_name(args.cam)
os.makedirs("calib", exist_ok=True)

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

obj_pts, img_pts, last_t, last_c, size, tries = [], [], 0.0, None, None, 0
PTS_F = f"calib/{NAME}_pts.npz"
if args.resolve:
    d = np.load(PTS_F)
    img_pts = [c.astype(np.float32) for c in d["corners"]]
    obj_pts = [objp] * len(img_pts)
    size = tuple(int(v) for v in d["res"])
else:
    cap = open_capture(args.cam, W, H)
    assert cap is not None, f"카메라 {args.cam}을 열 수 없음"
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
if not args.resolve:
    print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
while len(img_pts) < args.n and not args.resolve:
    ok, frame = cap.read()
    if not ok:
        continue
    if size is None:
        size = (frame.shape[1], frame.shape[0])
        print(f"카메라 해상도 {size[0]}x{size[1]} (실행할 때도 같은 해상도·스트림을 써야 함)")
    if time.time() - last_t < 1.0:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), cv2.CALIB_CB_FAST_CHECK)
    sb = False
    if not found:   # 광각 렌즈로 격자가 크게 휘면 기본 방식이 놓친다 → 왜곡에 강한 SB 방식으로 다시 시도
        found, corners = cv2.findChessboardCornersSB(gray, (args.cols, args.rows), cv2.CALIB_CB_EXHAUSTIVE)
        sb = found
    tries += 1
    if tries % 10 == 0 and not img_pts:
        print(f"  아직 인식 못 함 ({tries}회 시도). 체커보드가 화면의 1/3~1/2 크기로, 코너가 모두 보이게")
    if not found:
        continue
    c = corners.reshape(-1, 2).mean(0)
    if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치는 건너뜀
        continue
    if not sb:      # SB 방식은 이미 정밀한 코너를 준다
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp); img_pts.append(corners)
    last_t, last_c = time.time(), c
    cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)
    cv2.imwrite(f"calib/{NAME}_last.jpg", frame)
    print(f"  {len(img_pts)}/{args.n}")

if not args.resolve:
    np.savez(PTS_F, corners=np.stack(img_pts), res=np.array(size))   # 다시 촬영하지 않고 재계산할 수 있게 코너 저장
K0, flags = None, 0
if args.fix_center:
    K0 = np.array([[float(size[0]), 0, (size[0] - 1) / 2], [0, float(size[0]), (size[1] - 1) / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, K0, None, flags=flags)
np.savez(f"calib/{NAME}_intr.npz", K=K, dist=dist, res=np.array(size))
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/{NAME}_intr.npz")
