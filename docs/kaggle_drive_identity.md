# Kaggle ↔ Google Drive 데이터 동일성 검증

검증 대상: [Korean Product Expiration Date Detection Dataset](https://www.kaggle.com/datasets/kimhyeongminkhu/korean-expiry-date-ocr)

## 확인 결과

결론은 **Drive의 `상품사진입니다` 폴더가 Kaggle archive 안의
`expiry_region_detection_dataset/images/train` 이미지 집합과 동일한 파일
집합**이라는 것이다.

| 항목 | 결과 |
|---|---:|
| Drive 이미지 수 | 3,352 |
| Kaggle region train 이미지 수 | 3,352 |
| 파일명 교집합 | 3,343 |
| 파일명이 다른 이미지 | 9 |
| 파일명이 다른 9장의 내용 CRC32 일치 | 9/9 |
| 실제 Drive 샘플 CRC32가 Kaggle archive image CRC와 일치 | 232/232 |
| 전체 3,352장의 이름+파일크기 대응 | 3,343/3,343 + renamed 9/9 |

## 파일명 차이

Drive의 `3344.jpg`~`3352.jpeg` 일부가 Kaggle archive에서는 다음과 같이
원본 카메라 파일명으로 저장되어 있다.

```text
3344.jpg  ↔ KakaoTalk_20251101_124213882_06.jpg
3345.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-15-52 015.jpeg
3346.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-15-52 016.jpeg
3347.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-16-30 023.jpeg
3348.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-16-31 024.jpeg
3349.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-16-58 020.jpeg
3350.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-16-58 021.jpeg
3351.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-17-01 029.jpeg
3352.jpeg ↔ KakaoTalk_Photo_2025-12-09-15-17-01 030.jpeg
```

이 9장은 모두 이미지 바이트 CRC32가 일치한다. 나머지 3,343장은 파일명과
파일크기가 모두 일치하며, 그중 실제로 내려받은 표본 232장도 CRC32가
일치한다. 따라서 데이터 집합 구성 동일성은 확정적이다. 다만 3,352장
전부를 다시 다운로드해 바이트 단위 CRC를 계산한 것은 아니므로, “전수
바이트 해시 완료”라고 과장하지 않는다.

## 중요한 구분

Kaggle archive 전체에는 `expiry_date_detection_dataset`,
`expiry_region_detection_dataset`, `expiry_dmy_detection_dataset` 세 단계가
함께 들어 있다. 따라서 Kaggle archive 전체 이미지 수는 Drive보다 많지만,
대회에서 사용하는 Drive 3,352장은 그중 **region detection train subset**이다.

즉 “Drive와 Kaggle이 완전히 별개인가?”에 대한 답은 **아니다**이며,
현재 Drive 데이터는 Kaggle 데이터셋의 동일한 region-train 이미지 집합이다.
