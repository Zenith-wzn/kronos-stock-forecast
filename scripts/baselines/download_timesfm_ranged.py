from __future__ import annotations
import hashlib, os, time
from pathlib import Path
import requests
URL='https://huggingface.co/google/timesfm-3.0-pytorch/resolve/43046b85ec22d584a13f8098c2ed39c889e129c2/model.safetensors?download=true'
OUT=Path(__file__).resolve().parents[2]/'timesfm3/weights/model.safetensors'
PART=OUT.with_suffix('.safetensors.part')
SIZE=1322898824
CHUNK=16*1024*1024

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    n=PART.stat().st_size if PART.exists() else 0
    if n>SIZE: raise ValueError('Oversized partial file; preserve for diagnosis')
    with PART.open('ab') as f:
      while n<SIZE:
        end=min(n+CHUNK-1,SIZE-1)
        for attempt in range(20):
          try:
            with requests.get(URL,headers={'Range':f'bytes={n}-{end}'},stream=True,timeout=(30,180),allow_redirects=True) as r:
              r.raise_for_status()
              if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {n}-{end}/{SIZE}':
                raise IOError('Server did not honor requested byte range')
              data=b''.join(r.iter_content(1024*1024))
            expected=end-n+1
            if len(data)!=expected: raise IOError(f'partial {len(data)} != {expected}')
            f.write(data); f.flush(); os.fsync(f.fileno()); n=end+1
            print(f'{n}/{SIZE} ({n/SIZE:.1%})',flush=True); break
          except Exception as e:
            print(f'retry offset={n} attempt={attempt+1}: {e}',flush=True); time.sleep(min(30,2+attempt))
        else: raise RuntimeError(f'failed at {n}')
    digest=hashlib.sha256(PART.read_bytes()).hexdigest()
    print('sha256',digest,flush=True)
    expected='a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
    if digest!=expected: raise ValueError('hash mismatch')
    os.replace(PART,OUT)
if __name__=='__main__': main()
