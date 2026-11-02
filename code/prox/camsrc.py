"""10단계 공통: 카메라 입력을 같은 방식으로 연다 (교육장 장비 대응 추가 파일)
카메라 지정 방법:
  vigi            Jetson 유선 포트의 VIGI 카메라 (주소 VIGI_IP, 스트림 VIGI_STREAM, 비밀번호 CAM_PW 환경 변수)
  rtsp://...      다른 RTSP 카메라 주소
  0, 2, ...       USB 웹캠 장치 번호 (/dev/video0 등)
카메라 주소가 바뀌면: export VIGI_IP=10.42.0.xxx  (ip neigh show dev enP8p1s0 로 확인)
"""
import os, re
from urllib.parse import quote
import cv2

VIGI_IP = os.environ.get("VIGI_IP", "10.42.0.112")
VIGI_STREAM = os.environ.get("VIGI_STREAM", "stream2")   # stream2 = 848x480. 보정과 실행에 같은 스트림을 써야 함

def cam_name(spec):
    """보정 파일 이름에 쓸 짧은 이름 (vigi, 0, 2 ...)"""
    return re.sub(r"[^0-9A-Za-z]+", "_", spec).strip("_")[:24] or "cam"

def open_capture(spec, w=640, h=480):
    """열리면 cv2.VideoCapture, 못 열면 None. USB 웹캠일 때만 w, h(MJPG)를 요청한다"""
    if spec == "vigi" or spec.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        if spec == "vigi":
            if "CAM_PW" not in os.environ:
                raise SystemExit('CAM_PW가 없음: read -s -p "카메라 관리자 비밀번호: " CAM_PW; echo; export CAM_PW')
            spec = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{VIGI_IP}:554/{VIGI_STREAM}"
        cap = cv2.VideoCapture(spec, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(int(spec), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # 초점이 바뀌면 렌즈 보정값이 어긋나므로 고정 (C920 등)
        cap.set(cv2.CAP_PROP_FOCUS, 0)       # 0 = 먼 곳에 맞춤
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap if cap.isOpened() else None
