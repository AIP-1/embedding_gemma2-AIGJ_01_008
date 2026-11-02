"""모든 프레임의 손목 좌우 위치로, 흔드는 폭 기준을 바꿔 가며 1.5초 창마다 방향 전환 횟수를 센다"""
import sys, cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, body

cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS) or 15
det = YOLO("yolo11n-pose.pt"); hist = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1
    r = det(f, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0: continue
    k = np.concatenate([r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy()[:, None]], -1)
    b = body(k)
    if b is None: continue
    (cx, cy), sw = b
    w = LW if k[LW, 1] < k[RW, 1] else RW
    if k[w, 2] > KP_CONF and (cy - k[w, 1]) / sw > 0.3:
        hist.append((i / fps, (k[w, 0] - cx) / sw))

def changes(xs, th):
    n, last, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > th:
            cur = 1 if d > 0 else -1
            if last and cur != last: n += 1
            last, anchor = cur, x
    return n

print(f"들어올린 프레임 {len(hist)}개 / 전체 {i}개 ({fps:.0f} fps)")
for th in [0.15, 0.10, 0.08, 0.05, 0.03]:
    best, hit = 0, 0
    for t0, _ in hist:
        xs = [x for t, x in hist if t0 <= t < t0 + 1.5]
        if len(xs) >= 8:
            c = changes(xs, th); best = max(best, c); hit += c >= 2
    print(f"폭 기준 {th:.2f}: 1.5초 창 최대 방향 전환 {best}회, 손흔들기로 판정된 창 {hit}개")
