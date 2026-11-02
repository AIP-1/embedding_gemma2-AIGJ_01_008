#!/bin/bash
# 6단계: 구성마다 tegrastats를 기록하며 실행하고, 실행 전 RAM과 실행 중 최고 RAM·GPU 사용률을 출력
run() {
  name="$1"; shift
  base=$(free -m | awk '/Mem:/{print $3}')
  tegrastats --interval 500 > /tmp/ts_$$.log 2>/dev/null &
  tp=$!
  HF_HUB_OFFLINE=1 "$@" > /tmp/run_$$.log 2>&1
  kill $tp 2>/dev/null; wait $tp 2>/dev/null
  ram=$(grep -oE 'RAM [0-9]+' /tmp/ts_$$.log | awk '{print $2}' | sort -n | tail -1)
  gpu=$(grep -oE 'GR3D_FREQ [0-9]+' /tmp/ts_$$.log | awk '{print $2}' | sort -n | tail -1)
  printf "%-12s 실행 전 %5s MB | 최고 %5s MB (+%s) | GPU 최고 %3s%%\n" "$name" "$base" "$ram" "$((ram-base))" "$gpu"
}
run "텍스트"      python test_text.py
run "+비전 이미지" python test_vision.py
run "+비전 동영상" python test_video.py
run "+오디오"      python test_audio.py
run "전체"        python test_all.py
