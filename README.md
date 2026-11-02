# EmbeddingGemma 2 멀티모달 임베딩 실습

AIGJ_01_008 박민수. Jetson Orin Nano Super에서 EmbeddingGemma 2 설치부터 미세 조정, 카메라 물체 검색, 동작 인식, 접근 감지까지 진행한 결과다.

결과는 [REPORT.md](REPORT.md)에 있다.

| 폴더 | 내용 |
| --- | --- |
| `code/` | 1~9단계 코드 (`eg2.py`, `test_*.py`, `eval_ir.py`, `train_ft.py`, `make_data.py`, `app.py`, `pose_app.py`, `train_pose.py` 등) |
| `code/prox/` | 10단계 접근 감지와 보정 코드 |
| `data/` | 7단계 미세 조정 데이터 (`train.jsonl` 282쌍, `eval.jsonl` 60쌍) |

카메라 비밀번호는 환경 변수 `CAM_PW`로 입력하며 저장소에 넣지 않는다. 사진, 영상, 모델 파일도 넣지 않았다.
