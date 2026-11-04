"""9-4 빈칸 가상 시험: 합성 관절로 턱 괴기(raw)와 악수(raw·smooth)를 센다

pose_app.py의 판정 함수(pose_label, near_face, wave_wrist, is_wave, majority, handshake_pairs)를 원문 그대로 불러,
앱의 동작 루프와 같은 순서(raw 라벨 → 0.6초 다수결 → 손흔들기 → 라벨이 바뀌면 이벤트)로 돌린다. 카메라·YOLO는 쓰지 않는다.
합성 조건: 1280x720, 15fps, 어깨 너비 80px, 관절 흔들림 2px(팔 1.7배), 손목 신뢰도가 잠깐 떨어지는 프레임 3%.
사용법: python pose_sim94.py
"""
import collections
import numpy as np
from pose_model import KP_CONF, NOSE, LS, RS, LW, RW, body

src = open("pose_app.py", encoding="utf-8").read()
def grab(name):
    i = src.index(f"def {name}(")
    j = min(k for k in (src.find("\ndef ", i + 1), src.find("\nclass ", i + 1)) if k > 0)
    return src[i:j]
G = {"np": np, "collections": collections, "KP_CONF": KP_CONF, "NOSE": NOSE, "LS": LS, "RS": RS, "LW": LW, "RW": RW,
     "LE": 7, "RE": 8, "body": body, "pairs": {}, "FACE_R": 0.5}
import re, os
# 악수 기준값(HS_DIST 등)은 Jetson 앱 파일(pose_app.py, pose_model.py)에 적힌 값을 그대로 쓴다. 못 찾으면 완화 후 값
HS_DEFAULT = {"HS_DIST": 1.0, "HS_LOW": 4.0, "HS_CONF": 0.3}
srcs = src + "".join(open(f, encoding="utf-8").read() for f in ("pose_model.py",) if os.path.exists(f))
def const(name):
    # "HS_DIST, HS_LOW, HS_CONF = 1.0, 4.0, 0.3" 같은 여러 이름 한 줄 대입도 그대로 실행해 읽는다
    for line in re.findall(r"^[ \t]*[A-Z_, ]*\b%s\b[A-Z_, ]*=.*$" % name, srcs, re.M):
        env = {"os": os}
        try:
            exec(line.strip(), env)
            return float(env[name])
        except Exception:
            pass
    return HS_DEFAULT[name]
def call(fn, *a):
    while True:
        try:
            return G[fn](*a)
        except NameError as e:
            name = re.search(r"name '(\w+)'", str(e)).group(1)
            G[name] = const(name)
            print(f"  ({name} = {G[name]})")
for fn in ("near_face", "pose_label", "wave_wrist", "is_wave", "majority", "handshake_pairs"):
    exec(grab(fn), G)
SMOOTH_S, WAVE_S, FPS = 0.6, 1.5, 15.0
DT = 1 / FPS

def skeleton(cx, cy, sw, lw=None, rw=None, le=None, re=None):
    """정면 사람 17관절 (x, y, conf). 손목·팔꿈치 위치를 주면 그 자세"""
    k = np.zeros((17, 3)); k[:, 2] = 0.9
    pts = {0: (cx, cy - 0.9 * sw), 1: (cx - 0.12 * sw, cy - 1.0 * sw), 2: (cx + 0.12 * sw, cy - 1.0 * sw),
           3: (cx - 0.3 * sw, cy - 0.9 * sw), 4: (cx + 0.3 * sw, cy - 0.9 * sw),
           5: (cx - sw / 2, cy), 6: (cx + sw / 2, cy),
           7: le or (cx - 0.6 * sw, cy + 0.8 * sw), 8: re or (cx + 0.6 * sw, cy + 0.8 * sw),
           9: lw or (cx - 0.62 * sw, cy + 1.5 * sw), 10: rw or (cx + 0.62 * sw, cy + 1.5 * sw),
           11: (cx - 0.35 * sw, cy + 1.7 * sw), 12: (cx + 0.35 * sw, cy + 1.7 * sw),
           13: (cx - 0.38 * sw, cy + 2.6 * sw), 14: (cx + 0.38 * sw, cy + 2.6 * sw),
           15: (cx - 0.4 * sw, cy + 3.5 * sw), 16: (cx + 0.4 * sw, cy + 3.5 * sw)}
    for i, p in pts.items():
        k[i, :2] = p
    return k

def jitter(k, rng):
    k = k.copy()
    k[:, :2] += rng.normal(0, 2.0, (17, 2))
    k[7:11, :2] += rng.normal(0, 2.0 * 0.7, (4, 2))          # 팔은 1.7배
    for w in (9, 10):
        if rng.random() < 0.03:
            k[w, 2] = 0.2
    return k

def run_person(frames, mode):
    """frames: [(t, k)]. 앱의 update_person 동작 부분과 같은 순서로 이벤트를 센다"""
    st = {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90), "label": "기본"}
    ev = collections.Counter()
    for now, k in frames:
        b = body(k)
        if b is None:
            continue
        raw = G["pose_label"](k, b)
        st["raw"].append((now, raw))
        w = G["wave_wrist"](k, b)
        if w is not None:
            st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
        if mode == "raw":
            label = raw
        else:
            v, frac = G["majority"](st["raw"], now, SMOOTH_S)
            label = v if frac >= 0.6 else st["label"]
            recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
            still_up = any(now - t <= 0.5 for t, _ in recent)
            if label in ("기본", "한팔들기") and (G["is_wave"](recent) or (st["label"] == "손흔들기" and still_up)):
                label = "손흔들기"
        if label != st["label"]:
            if label != "기본":
                ev[label] += 1
            st["label"] = label
    return ev

def chin_frames(rng, reps=3, hold=3.0, rest=2.0):
    """턱 괴기: 오른 손목을 턱 아래로, 팔꿈치는 가슴 앞. 0.4초에 걸쳐 올리고 내린다"""
    cx, cy, sw = 640, 300, 80
    rest_w, rest_e = (cx + 0.62 * sw, cy + 1.5 * sw), (cx + 0.6 * sw, cy + 0.8 * sw)
    chin_w, chin_e = (cx + 0.1 * sw, cy - 0.55 * sw), (cx + 0.35 * sw, cy + 0.35 * sw)
    out, t = [], 0.0
    for _ in range(reps):
        for phase, secs in (("up", 0.4), ("hold", hold), ("down", 0.4), ("rest", rest)):
            for i in range(int(secs * FPS)):
                f = {"up": (i + 1) / (0.4 * FPS), "hold": 1, "down": 1 - (i + 1) / (0.4 * FPS), "rest": 0}[phase]
                w = tuple(np.add(rest_w, f * np.subtract(chin_w, rest_w)))
                e = tuple(np.add(rest_e, f * np.subtract(chin_e, rest_e)))
                out.append((t, jitter(skeleton(cx, cy, sw, rw=w, re=e), rng))); t += DT
    return out

def handshake(rng, mode, reps=3, hold=3.0, rest=2.0, clasp_drop=0.25):
    """두 사람이 마주 보고 악수. 맞잡은 동안 손을 위아래로 흔들고(2Hz, 어깨 너비 0.3배),
    두 손목이 서로 가려 신뢰도가 떨어지는 프레임이 clasp_drop 비율로 생긴다. 앱의 2인 루프와 같은 방식으로 센다"""
    G["pairs"].clear()
    ca, cb, cy, sw = 560, 720, 300, 80
    meet = ((ca + cb) / 2, cy + 1.0 * sw)
    on_set, n_ev, t = set(), 0, 0.0
    for _ in range(reps):
        for phase, secs in (("in", 0.5), ("hold", hold), ("out", 0.5), ("rest", rest)):
            for i in range(int(secs * FPS)):
                f = {"in": (i + 1) / (0.5 * FPS), "hold": 1, "out": 1 - (i + 1) / (0.5 * FPS), "rest": 0}[phase]
                shake = 0.3 * sw * np.sin(2 * np.pi * 2 * t) if phase == "hold" else 0
                a_rest, b_rest = (ca + 0.62 * sw, cy + 1.5 * sw), (cb - 0.62 * sw, cy + 1.5 * sw)
                aw = (a_rest[0] + f * (meet[0] - 4 - a_rest[0]), a_rest[1] + f * (meet[1] - a_rest[1]) + shake)
                bw = (b_rest[0] + f * (meet[0] + 4 - b_rest[0]), b_rest[1] + f * (meet[1] - b_rest[1]) + shake)
                A = jitter(skeleton(ca, cy, sw, rw=aw), rng); B = jitter(skeleton(cb, cy, sw, lw=bw), rng)
                if phase == "hold" and rng.random() < clasp_drop:
                    A[10, 2] = 0.3; B[9, 2] = 0.3
                found = call("handshake_pairs", [A, B], [1, 2], t)
                for key in list(G["pairs"]):
                    if key not in found:
                        G["pairs"].pop(key); on_set.discard(key); continue
                    if mode == "raw":
                        on = G["pairs"][key][-1][1]
                    else:
                        v, frac = G["majority"](G["pairs"][key], t, SMOOTH_S)
                        on = bool(v) and frac >= 0.6
                    if on and key not in on_set:
                        n_ev += 1
                    (on_set.add if on else on_set.discard)(key)
                t += DT
    return n_ev

print("합성 조건: 15fps, 어깨 너비 80px, 관절 흔들림 2px, 3회씩, 시드 3개 평균\n")
rows = []
for name, fr in (("턱 괴기 (FACE_R=0, 문서 기본값)", 0.0), ("턱 괴기 (얼굴 근처 제외, 개선 후)", 0.5)):
    G["FACE_R"] = fr
    r = [run_person(chin_frames(np.random.default_rng(s)), "raw") for s in range(3)]
    sm = [run_person(chin_frames(np.random.default_rng(s)), "smooth") for s in range(3)]
    fmt = lambda cs: ", ".join(f"{k} {np.mean([c[k] for c in cs]):.1f}" for k in sorted({k for c in cs for k in c})) or "없음"
    print(f"| {name} | raw: {fmt(r)} | smooth: {fmt(sm)} |")
G["FACE_R"] = 0.5
hr = [handshake(np.random.default_rng(s), "raw") for s in range(3)]
hs = [handshake(np.random.default_rng(s), "smooth") for s in range(3)]
print(f"| 악수 (실제 3회) | raw: 악수 {np.mean(hr):.1f} (회차별 {hr}) | smooth: 악수 {np.mean(hs):.1f} (회차별 {hs}) |")
