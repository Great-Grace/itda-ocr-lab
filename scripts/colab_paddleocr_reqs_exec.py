from importlib.metadata import metadata
for req in metadata('paddleocr').get_all('Requires-Dist') or []:
 print(req)
