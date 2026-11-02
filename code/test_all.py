"""전체 모델: 이미지·동영상·소리를 한 인덱스에 넣고 텍스트 질의 하나로 검색"""
import glob, time
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
        img.thumbnail((max_side, max_side))
        frames.append(np.asarray(img))
        next_t += 1 / fps
    c.close()
    return np.stack(frames)

t = time.perf_counter()
model = load(vision=True, audio=True)
print(f"전체 모델 로드: {(time.perf_counter()-t):.1f} 초")

names, inputs = [], []
for p in sorted(glob.glob("media/*.jpg")):
    names.append("[이미지] " + p.split("/")[-1]); inputs.append({"image": p})
names.append("[동영상] clip.mp4"); inputs.append({"video": read_frames("media/clip.mp4")})
for p in sorted(glob.glob("media/*.wav")):
    names.append("[소리] " + p.split("/")[-1]); inputs.append({"audio": p})

t = time.perf_counter()
emb = np.stack([model.encode(x, normalize_embeddings=True) for x in inputs])
print(f"혼합 {len(inputs)}개 인덱싱: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["칠판이 있는 벽", "a photo of a person", "강의실을 촬영한 영상",
           "날카로운 경보음", "에어컨을 켜 달라는 목소리", "바닷가 해변"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
for qi, row in zip(queries, q @ emb.T):
    order = row.argsort()[::-1][:3]
    print(f"{qi} -> " + ", ".join(f"{names[i]} ({row[i]:.3f})" for i in order))
