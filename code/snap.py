"""RTSP 카메라에서 지금 장면 한 장을 media/<이름>.jpg로 저장. 사용법: python snap.py 칠판"""
import os, sys, av
from urllib.parse import quote

name = sys.argv[1] if len(sys.argv) > 1 else "snap"
url = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@10.42.0.112:554/stream1"
c = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
for i, f in enumerate(c.decode(c.streams.video[0])):
    if i == 10:                      # 앞 프레임은 화질이 불안정할 수 있어 몇 장 건너뜀
        f.to_image().save(f"media/{name}.jpg")
        break
c.close()
print("저장:", f"media/{name}.jpg")
