import base64
import json
import urllib.request
from pathlib import Path

import os
api_key = os.environ.get("GEMINI_API_KEY", "")
img_dir = Path("/content/drive/MyDrive/상품사진입니다")
img_path = next(img_dir.glob("*.jpg"))
print("Testing on real image:", img_path.name, "Size:", img_path.stat().st_size)

b64_data = base64.b64encode(img_path.read_bytes()).decode('utf-8')
prompt = """이 상품 사진에서 유통기한 또는 소비기한 날짜를 찾아줘.
반드시 YYYY-MM-DD 형식으로 작성하고, 다음과 같은 JSON 형식으로만 응답해:
{"date": "YYYY-MM-DD", "text_found": "사진에서 찾은 원본 텍스트", "confidence": "high|medium|low", "reason": "선택 이유"}"""

payload = {
    'contents': [{
        'parts': [
            {'text': prompt},
            {
                'inline_data': {
                    'mime_type': 'image/jpeg',
                    'data': b64_data
                }
            }
        ]
    }],
    'generationConfig': {
        'response_mime_type': 'application/json',
        'temperature': 0.0
    }
}

for model in ['gemini-3.6-flash', 'gemini-flash-latest', 'gemini-2.5-flash']:
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"=== [{model} SUCCESS!] ===")
            print(data['candidates'][0]['content']['parts'][0]['text'])
            break
    except urllib.error.HTTPError as e:
        print(f"{model} failed: {e.code} {e.read().decode('utf-8')[:150]}")
    except Exception as e:
        print(f"{model} failed: {e}")
