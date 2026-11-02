"""영상의 프레임별 손목 높이·좌우 위치를 출력해 손 흔들기 판정 기준과 비교"""
import sys, cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, body

cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS) or 15
det = YOLO("yolo11n-pose.pt"); i = 0
print(f"{fps:.0f} fps. 높이 = (어깨 - 손목)/어깨너비, 0.3 이상이면 '들어 올림'. 좌우 = 손목 x/어깨너비")
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1
    r = det(f, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0: continue
    k = np.concatenate([r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy()[:, None]], -1)
    b = body(k)
    if b is None:
        print(f"{i/fps:5.1f}s  어깨 안 보임"); continue
    (cx, cy), sw = b
    w = LW if k[LW, 1] < k[RW, 1] else RW
    h = (cy - k[w, 1]) / sw; x = (k[w, 0] - cx) / sw
    if i % max(1, int(fps / 5)) == 0:   # 0.2초마다 출력
        print(f"{i/fps:5.1f}s  높이 {h:+.2f} {'들어올림' if h > 0.3 and k[w,2] > KP_CONF else '-':6}  좌우 {x:+.2f}  신뢰도 {k[w,2]:.2f}")
