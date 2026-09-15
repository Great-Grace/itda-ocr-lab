import subprocess
subprocess.run([
    'tar', '-czf', '/content/ppocrv6_medium_rec.tar.gz',
    '-C', '/root/.paddlex/official_models', 'PP-OCRv6_medium_rec'
], check=True)
print('B4_WEIGHTS_ARCHIVE_READY')
