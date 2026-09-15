# Recognizer research options

## Evidence from actual experiments

The real Drive dev/lock comparison currently supports `PP-OCRv6_medium_rec`:

- dev100: 69.0% EM, 74.2% Candidate Coverage
- lock100: 66.0% EM, 71.0% Candidate Coverage
- CPU: about 1.8 sec/image, 78MB model payload

## Why server_rec is not an immediate drop-in

The official PaddleOCR documentation lists `PP-OCRv5_server_rec` as a large
general recognizer, but its default language coverage is not Korean. The same
documentation describes Korean support through dictionary extension and
fine-tuning rather than a ready-made Korean server weight.

Therefore the next server experiment must be one of:

1. Obtain or train a Korean server-rec dictionary-compatible weight.
2. Use a real Korean/date crop dataset to fine-tune the server model.
3. Compare it against the already verified V6 medium model under identical CPU
   latency/RAM gates.

Blindly substituting the server model into the current Korean pipeline would be
an invalid architecture comparison because the output dictionary would differ.
