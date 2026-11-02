"""영상에서 0.5초마다 사람 수, 어깨 보임 여부, 두 사람 손목 최소 거리와 높이 조건을 출력"""
import sys, cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, body

cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS) or 15
det = YOLO("yolo11n-pose.pt"); i = 0
print(f"{fps:.0f} fps. 거리 = 두 사람 손목 최소 거리 / 평균 어깨너비 (0.5 미만이면 악수 후보)")
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1
    if i % max(1, int(fps / 2)): continue
    r = det(f, verbose=False)[0]
    n = 0 if r.keypoints is None else len(r.keypoints.xy)
    if n < 2:
        print(f"{i/fps:5.1f}s  사람 {n}명"); continue
    kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
    a, b = kps[0], kps[1]
    ba, bb = body(a), body(b)
    if ba is None or bb is None:
        print(f"{i/fps:5.1f}s  사람 {n}명, 어깨 보임: A {'O' if ba else 'X'} / B {'O' if bb else 'X'}"); continue
    sw = (ba[1] + bb[1]) / 2; best, hgt = 1e9, None
    for wa in (LW, RW):
        for wb in (LW, RW):
            if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                d = np.linalg.norm(a[wa, :2] - b[wb, :2])
                if d < best:
                    best = d; mid_y = (a[wa, 1] + b[wb, 1]) / 2
                    hgt = (mid_y - ba[0][1]) / ba[1]
    if hgt is None:
        print(f"{i/fps:5.1f}s  사람 {n}명, 어깨 OK, 손목 안 보임"); continue
    ok_h = 0 < hgt < 2.5
    print(f"{i/fps:5.1f}s  사람 {n}명, 거리 {best/sw:.2f} {'가까움' if best < 0.5*sw else '-':4}  높이 {hgt:+.2f} {'OK' if ok_h else '범위밖'}")
