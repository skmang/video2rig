"""下载已授权的 H3 首帧生成组件；固定版本、断点续传并核对官方 SHA256。"""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--models-dir', required=True, type=Path)
parser.add_argument('--cache-dir', required=True, type=Path)
args = parser.parse_args()
ROOT = args.cache_dir.resolve()
MODELS = args.models_dir.resolve()
REVISION = 'e5eb578a89295337b8ff433a035929ce0279e0b6'
FILES = [
    ('diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors', 15983746636, 'ac746a2e41628ab25afd44d2b22a7fab8d7e66cd07a01bf89ed9d74b0d3f0c35'),
    ('text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 15687142551, '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'),
    ('vae/minimax_h3_video_vae_int8_convrot.safetensors', 2811065184, '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'),
    ('vae/minimax_h3_audio_vae_fp32.safetensors', 605254808, '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
]
CHUNK = 128 * 1024 * 1024
SEGMENTS = ROOT / 'segments'
SEGMENTS.mkdir(parents=True, exist_ok=True)
manifest = {'repository': 'Comfy-Org/MiniMax-H3', 'revision': REVISION, 'files': []}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def fetch(job):
    remote, start, stop, path = job
    for attempt in range(6):
        offset = path.stat().st_size if path.exists() else 0
        if offset == stop - start:
            return
        if offset > stop - start:
            raise RuntimeError(f'Unexpected segment length: {path}')
        url = f'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{REVISION}/{remote}?segment={start}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={start + offset}-{stop - 1}'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start + offset}-{stop - 1}/'):
                    raise RuntimeError('Server did not honor the requested byte range')
                with path.open('ab') as output:
                    while offset < stop - start:
                        data = response.read(min(1024 * 1024, stop - start - offset))
                        if not data:
                            raise OSError('Incomplete response')
                        output.write(data)
                        offset += len(data)
            return
        except Exception as error:
            print('RETRY', remote, start, attempt + 1, str(error), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))


for remote, size, expected in FILES:
    destination = MODELS / remote
    record = {'file': remote, 'bytes': size, 'sha256': expected, 'path': str(destination)}
    if destination.exists():
        if destination.stat().st_size != size or sha256(destination) != expected:
            raise RuntimeError(f'Existing destination differs; refusing to overwrite {destination}')
    else:
        jobs = [(remote, start, min(size, start + CHUNK), SEGMENTS / f'{Path(remote).name}.{start}.part') for start in range(0, size, CHUNK)]
        started = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            while not all(f.done() for f in futures):
                count = sum(j[3].stat().st_size for j in jobs if j[3].exists())
                print(f'DOWNLOAD {Path(remote).name} {count / 1e9:.2f}/{size / 1e9:.2f} GB {time.time() - started:.0f}s', flush=True)
                wait(futures, timeout=20)
            for future in futures:
                future.result()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.safetensors.download')
        digest = hashlib.sha256()
        with temporary.open('wb') as output:
            for _, _, _, path in jobs:
                with path.open('rb') as source:
                    while data := source.read(8 * 1024 * 1024):
                        digest.update(data)
                        output.write(data)
        if temporary.stat().st_size != size or digest.hexdigest() != expected:
            raise RuntimeError(f'Checksum failed: {temporary}')
        temporary.rename(destination)
    record['status'] = 'verified'
    manifest['files'].append(record)
    (ROOT / 'download-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('VERIFIED', destination, expected, flush=True)
print('ALL_MODELS_VERIFIED', flush=True)
