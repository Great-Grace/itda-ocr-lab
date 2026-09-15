# Colab 데이터 실행 루틴

실제 상품 이미지 실험은 아래 순서를 고정한다.

1. 새 세션을 매번 만들지 않고, 하나의 `itda-data-*` 세션을 만든다.
2. `scripts/colab_data_preflight.py --mount`를 한 번만 실행한다.
3. OAuth가 필요하면 브라우저에서 한 번 승인한다. 실패하면 재시도하지 않고 세션 상태를 보존한다.
4. preflight가 `상품사진입니다` 폴더, 3,352장, detector/recognizer weight 디렉터리를 모두 확인한 뒤에만 실험을 시작한다.
5. 실험 스크립트는 `/content/itda_drive_preflight.json`의 경로만 사용한다.
6. 모든 실험이 끝난 뒤에만 세션을 중지한다.

Synthetic architecture 학습은 Drive mount가 필요 없으므로 별도 GPU 세션에서 실행한다.
