"""녹화 CSV를 세션(녹화 묶음)별로 보여 주고, 잘못된 라벨 파일에 들어간 세션을 다른 파일로 옮긴다
사용법:
  python split_sessions.py data/action/idle.csv                          # 세션 목록
  python split_sessions.py data/action/idle.csv move 3 data/action/clap.csv   # 3번 세션을 clap.csv 뒤에 붙이고 idle.csv에서 뺌
처음 옮길 때 원본은 <파일>.bak 으로 한 번 백업한다.
"""
import sys, os, csv, shutil, datetime

GAP_S = 5.0   # 이보다 오래 끊기면 다른 녹화로 본다

def sessions(path):
    rows = [r for r in csv.reader(open(path)) if r]
    rows.sort(key=lambda r: float(r[0]))
    out, cur = [], []
    for r in rows:
        if cur and float(r[0]) - float(cur[-1][0]) > GAP_S:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out

src = sys.argv[1]
ss = sessions(src)
if len(sys.argv) == 2:
    print(f"{src}: 세션 {len(ss)}개")
    for i, s in enumerate(ss, 1):
        t0, t1 = float(s[0][0]), float(s[-1][0])
        start = datetime.datetime.fromtimestamp(t0).strftime("%H:%M:%S")
        print(f"  {i}번  시작 {start}  길이 {t1 - t0:5.1f}초  {len(s)}줄")
    sys.exit()

assert sys.argv[2] == "move", "사용법: python split_sessions.py <원본.csv> move <번호> <대상.csv>"
idx, dst = int(sys.argv[3]), sys.argv[4]
assert 1 <= idx <= len(ss), f"세션 번호는 1~{len(ss)}"
if not os.path.exists(src + ".bak"):
    shutil.copy(src, src + ".bak")
with open(dst, "a", newline="") as f:
    csv.writer(f).writerows(ss[idx - 1])
with open(src, "w", newline="") as f:
    csv.writer(f).writerows(r for i, s in enumerate(ss, 1) if i != idx for r in s)
print(f"{idx}번 세션 {len(ss[idx - 1])}줄을 {dst} 로 옮김. {src} 에는 {sum(len(s) for i, s in enumerate(ss, 1) if i != idx)}줄 남음")
