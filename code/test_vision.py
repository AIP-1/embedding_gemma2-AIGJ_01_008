import glob, time
from eg2 import load

model = load(vision=True)

images = sorted(glob.glob("media/*.jpg"))
t = time.perf_counter()
img_emb = model.encode([{"image": p} for p in images], normalize_embeddings=True)
print(f"이미지 {len(images)}장: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["의자에 앉은 사람", "사람", "사람의 얼굴과 상체", "a person sitting", "a photo of a person"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

sim = q @ img_emb.T
for qi, row in zip(queries, sim):
    order = row.argsort()[::-1][:3]
    print(qi, "->", [(images[i].split('/')[-1], round(float(row[i]), 3)) for i in order])
