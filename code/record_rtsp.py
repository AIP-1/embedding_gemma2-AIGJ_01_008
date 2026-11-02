"""RTSP 카메라에서 10초를 media/clip.mp4로 저장하고, 2초마다 프레임을 media/frame_NN.jpg로 저장"""
import os, av
from urllib.parse import quote

IP, SECONDS, EVERY = "10.42.0.112", 10, 2
url = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{IP}:554/stream1"

inp = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
vin = inp.streams.video[0]
print("코덱", vin.codec_context.name, vin.codec_context.width, "x", vin.codec_context.height)
out = av.open("media/clip.mp4", "w")
vout = out.add_stream_from_template(vin)

t0, next_shot, n = None, 0.0, 0
for pkt in inp.demux(vin):
    if pkt.pts is None:
        continue
    if t0 is None:
        if not pkt.is_keyframe:          # 첫 키프레임부터 저장
            continue
        t0 = pkt.pts
    t = float((pkt.pts - t0) * vin.time_base)
    if t > SECONDS:
        break
    for frame in pkt.decode():
        if t >= next_shot:
            n += 1
            frame.to_image().save(f"media/frame_{n:02d}.jpg")
            print(f"  frame_{n:02d}.jpg ({t:.1f}초)")
            next_shot += EVERY
    pkt.pts -= t0
    pkt.dts = pkt.pts if pkt.dts is None else max(pkt.dts - t0, 0)
    pkt.stream = vout
    out.mux(pkt)

out.close(); inp.close()
print("저장 완료: media/clip.mp4, 프레임", n, "장")
