import json, urllib.request
try:
    r = json.load(urllib.request.urlopen("http://localhost:8001/recent", timeout=5))
except Exception as e:
    raise SystemExit(f"app.py가 아직 응답하지 않음 ({e})")
s = r["stats"]
print(f"[{r['mode']}] 검출 {s['fps']:.1f} fps, 임베딩 {s['embed_ms']:.0f} ms/개, 총 {r['total']}개")
for it in r["items"]:
    print(f"  {it['label']} ({it['lscore']})  <- YOLO: {it['cls']} {it['conf']}")
