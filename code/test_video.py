"""PyAV로 동영상을 초당 1장씩 읽어 임베딩 (torchcodec 없이). 실패하면 프레임 이미지 평균으로 대체"""
import time
import av, numpy as np
from eg2 import load

def read_frames(path, fps=1.0, max_side=640):
    c = av.open(path)
    s = c.streams.video[0]
    frames, next_t = [], 0.0
    for f in c.decode(s):
        t = float(f.pts * s.time_base)
        if t + 1e-6 < next_t:
            continue
        img = f.to_image()
        img.thumbnail((max_side, max_side))       # 2560x1440 → 640x360 (메모리 절약)
        frames.append(img)
        next_t += 1 / fps
    c.close()
    return frames

model = load(vision=True)
frames = read_frames("media/clip.mp4")
print(f"프레임 {len(frames)}장 읽음 ({frames[0].size[0]}x{frames[0].size[1]})")

t = time.perf_counter()
try:
    v = model.encode({"video": np.stack([np.asarray(f) for f in frames])}, normalize_embeddings=True)
    how = "동영상 입력"
except Exception as e:
    print("동영상 입력 실패 → 프레임 평균으로 대체:", type(e).__name__)
    e_img = model.encode([{"image": f} for f in frames], normalize_embeddings=True)
    v = e_img.mean(0)
    v = v / np.linalg.norm(v)
    how = "프레임 평균"
print(f"{how}: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["강의실 내부", "바닷가 해변"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
for qi, s in zip(queries, q @ v):
    print(f"  {qi}: {s:.3f}")
