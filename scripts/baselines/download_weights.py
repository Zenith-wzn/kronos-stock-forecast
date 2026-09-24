"""Download immutable official checkpoint snapshots and record provenance."""
from __future__ import annotations
import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cpu_baselines.common import atomic_json, read_json, sha256

SPECS = {
 'chronos2': ('amazon/chronos-2', '29ec3766d36d6f73f0696f85560a422f50e8498c'),
 'timesfm3': ('google/timesfm-3.0-pytorch', '43046b85ec22d584a13f8098c2ed39c889e129c2'),
}

def main():
    from huggingface_hub import snapshot_download, HfApi
    p = argparse.ArgumentParser(); p.add_argument('--model', choices=SPECS, required=True)
    a = p.parse_args(); home = ROOT / a.model; repo, revision = SPECS[a.model]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    required = [e for e in info.siblings if e.rfilename.endswith(('.json','.safetensors')) or e.rfilename in ('README.md','LICENSE')]
    if not all((home/'weights'/e.rfilename).is_file() for e in required):
        snapshot_download(repo, revision=revision, local_dir=str(home / 'weights'),
                          allow_patterns=['*.json', '*.safetensors', 'README.md', 'LICENSE'], max_workers=2)
    hashes = {}
    for entry in info.siblings:
        fp = home / 'weights' / entry.rfilename
        if fp.is_file():
            hashes[entry.rfilename] = sha256(fp)
            lfs = getattr(entry, 'lfs', None)
            expected = getattr(lfs, 'sha256', None) if lfs else None
            if expected and hashes[entry.rfilename] != expected:
                raise ValueError(f'Official LFS SHA256 mismatch: {fp}')
    commit = subprocess.check_output(['git', '-C', str(home/'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(home/'upstream'), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError(f'Official checkout not clean: {dirty}')
    distributions = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (home/'requirements.lock.txt').write_text('\n'.join(distributions)+'\n', encoding='utf-8')
    provenance = {'model_id': repo, 'revision': revision, 'source_commit': commit,
                  'weight_sha256': hashes, 'python': sys.version, 'dependencies': distributions,
                  'source_clean': True, 'device': 'cpu', 'dtype': 'float32'}
    atomic_json(home/'provenance.json', provenance)
    print(provenance, flush=True)

if __name__ == '__main__':
    main()
