"""RTSP 카메라 영상을 지정한 시간만큼 mp4로 저장 (재인코딩 없이 그대로 복사)
사용법: python rec_clip.py 60 media/pose_test.mp4 [stream2]
"""
import os, sys, av
from urllib.parse import quote

IP = "10.42.0.112"   # 카메라 주소가 바뀌면 ip neigh show dev enP8p1s0로 확인 후 수정
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
out_path = sys.argv[2] if len(sys.argv) > 2 else "media/pose_test.mp4"
stream = sys.argv[3] if len(sys.argv) > 3 else "stream2"   # pose_app.py 기본과 같은 저화질
url = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{IP}:554/{stream}"

inp = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
vin = inp.streams.video[0]
out = av.open(out_path, "w")
vout = out.add_stream_from_template(vin)

t0, last_sec = None, -1
print(f"녹화 시작: {seconds:.0f}초 → {out_path}")
for pkt in inp.demux(vin):
    if pkt.pts is None:
        continue
    if t0 is None:
        if not pkt.is_keyframe:          # 첫 키프레임부터 저장
            continue
        t0 = pkt.pts
    t = float((pkt.pts - t0) * vin.time_base)
    if t > seconds:
        break
    if int(t) != last_sec and int(t) % 5 == 0:
        last_sec = int(t)
        print(f"  {last_sec}초")
    pkt.pts -= t0
    pkt.dts = pkt.pts if pkt.dts is None else max(pkt.dts - t0, 0)
    pkt.stream = vout
    out.mux(pkt)

out.close(); inp.close()
print("저장 완료:", out_path)
