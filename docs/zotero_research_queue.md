# Zotero-grounded OCR experiment queue

This queue connects the local Zotero reading set to reproducible experiments in the ITDA OCR lab. The lock100 set remains frozen; new ideas are screened on screen32 and confirmed on dev100.

## Findings used

| Zotero item | Finding | Experiment implication |
|---|---|---|
| `QLPNWL59` — PP-OCRv3 (2022) | SVTR-LCNet replaces CRNN in the lightweight recognizer; height 48 improves recognition at a modest speed cost; GTC/TextConAug/TextRotNet/U-DML improve training but are removed or amortized at inference | Test recognition input height 32 vs 48; treat GTC and U-DML as training-only reproduction candidates |
| `XEBTJC2I` — SVTRv2 (2024) | CTC recognizer with multi-size resizing and feature rearrangement; semantic guidance is training-only and FRM remains at inference | Prototype a date-line SVTRv2-CTC recognizer after labeled crop generation |
| `Q4BVCBIF` — DBNet (2019) | Differentiable binarization detector is a strong detector alternative | Run only after recognition experiments; current detector timing is not the dominant error signal |
| `F8HXGKDZ` — ASTER (2019) | Rectification helps perspective/irregular text | Use as selective second-pass on uncertain date ROIs, not as the default full-image pass |
| `BBILPFXR` — CRNN (2015) | CTC sequence recognition handles variable-length strings without character boxes | Build a numeric/date CRNN baseline using synthetic date strings and cropped OCR regions |
| `53XEAKYC` — STR model comparison (2019) | Accuracy, speed, memory, training data and evaluation protocol must be aligned | Keep fresh-vs-cached runtime separate and never compare lock results to tuned dev results |

## Execution order

1. B4 input-height ablation: 32 vs 48, raw image, same detector.
2. B4 selective second-pass on low-confidence or no-candidate images.
3. B4 plus OCR character-repair/date parser variants using cached tokens.
4. Korean PP-OCRv3/v4 or SVTR-LCNet-compatible recognizer if an offline weight is available.
5. Synthetic numeric/date CRNN-CTC baseline.
6. SVTRv2-Tiny/Small CTC prototype with multi-size resizing and FRM.
7. DBNet detector comparison only if detector recall becomes limiting.
8. ASTER/TPS rectification only for the hard ROI subset.

## Stop/advance rules

- Reject a screen candidate if Candidate Recall or Final EM is below the current best by more than 5 percentage points.
- Promote to dev100 only when screen32 improves Candidate Recall and does not violate CPU/RAM constraints.
- Evaluate lock100 only after the recognizer, parser, selector, and runtime configuration are frozen.
- Keep a separate row for cached token experiments; cached runtime is not inference throughput.
