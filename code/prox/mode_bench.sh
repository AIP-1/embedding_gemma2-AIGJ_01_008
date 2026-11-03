#!/usr/bin/env bash
# 10-7 모드별 fps·카메라 시간 차·RAM: 모드마다 proximity_app.py를 SECS초 돌리며 /state와 프로세스 메모리를 기록한다
# 위치 보정과 무관한 값이라 카메라를 옮긴 뒤에도 잴 수 있다. 타겟은 좌표로 줘서 YOLOE·EmbeddingGemma를 올리지 않는다.
# 사용법 (~/eg2-lab/prox, venv와 CAM_PW 준비 뒤): bash mode_bench.sh 40
SECS=${1:-40}; PORT=8004; OUT=../runs/mode_bench_$(date +%H%M).csv
echo "mode,fps,skew_ms,rss_mb,avail_mb" > "$OUT"
run() {  # $1 = 모드 이름, 나머지 = proximity_app 인자
  local mode=$1; shift
  python proximity_app.py "$@" --target-xy 1.8,0 --port $PORT > /tmp/mode_$mode.log 2>&1 &
  local pid=$!
  sleep 25                                       # 모델 로드·카메라 연결
  for i in $(seq 1 $((SECS / 5))); do
    s=$(curl -s --max-time 2 localhost:$PORT/state)
    fps=$(echo "$s" | python3 -c "import sys,json; print(json.load(sys.stdin)['stats'].get('fps',''))" 2>/dev/null)
    skew=$(echo "$s" | python3 -c "import sys,json; print(json.load(sys.stdin)['stats'].get('skew_ms',''))" 2>/dev/null)
    rss=$(ps -o rss= -p $pid | awk '{print int($1/1024)}')
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$mode,$fps,$skew,$rss,$avail" >> "$OUT"
    sleep 5
  done
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
run ankle1 --mode ankle1 --cams 0 --calib 1
run stereo --mode stereo --cams vigi 0
run fused  --mode fused  --cams vigi 0
echo "저장: $OUT"
python3 - "$OUT" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
for m in ("ankle1", "stereo", "fused"):
    r = [x for x in rows if x["mode"] == m and x["fps"]]
    if not r:
        print(m, "기록 없음 (/tmp/mode_%s.log 확인)" % m); continue
    f = lambda k: [float(x[k]) for x in r if x[k] not in ("", "None")]
    print(f"| {m} | {st.median(f('fps')):.1f} | {st.median(f('skew_ms')) if f('skew_ms') else '-'} | {max(f('rss_mb')):.0f} |")
EOF
