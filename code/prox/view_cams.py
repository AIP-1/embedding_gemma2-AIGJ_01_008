"""10단계 설치 확인: 카메라 화면을 브라우저에서 나란히 본다 (방향·겹치는 구역 맞추기용, 저장 없음)
사용법: python view_cams.py --cams vigi 0   →  브라우저에서 http://<Jetson IP>:8001
"""
import argparse, threading, time
import cv2, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from camsrc import open_capture

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs="+", default=["vigi", "0"], help="vigi, rtsp://..., 또는 USB 장치 번호")
ap.add_argument("--res", default="640x480", help="USB 웹캠 요청 해상도")
ap.add_argument("--port", type=int, default=8001)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
jpegs = [None] * len(args.cams)

def cam_loop(n, spec):
    while True:
        cap = open_capture(spec, W, H)
        if cap is None:
            print(f"카메라 {spec}을 열 수 없음, 3초 뒤 재시도"); time.sleep(3); continue
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            cv2.line(f, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)    # 화면 중심 표시
            cv2.line(f, (0, h // 2), (w, h // 2), (0, 255, 255), 1)
            jpegs[n] = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        cap.release()
        print(f"카메라 {spec} 끊김, 재연결"); time.sleep(1)

app = FastAPI()

@app.get("/stream/{n}")
def stream(n: int):
    def gen():
        while True:
            f = jpegs[n]
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
def index():
    imgs = "".join(f'<div>{c}<br><img src="/stream/{i}"></div>' for i, c in enumerate(args.cams))
    return f"""<!doctype html><meta charset=utf-8><title>카메라 확인</title>
<style>body{{font-family:sans-serif;margin:16px;background:#111;color:#eee}}
.row{{display:flex;gap:12px;flex-wrap:wrap}}.row div{{flex:1;min-width:300px}}img{{width:100%;border-radius:6px}}</style>
<h2>카메라 확인</h2><div class=row>{imgs}</div>"""

if __name__ == "__main__":
    for n, spec in enumerate(args.cams):
        threading.Thread(target=cam_loop, args=(n, spec), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
