"""Offline inventory, checksum and Python-syntax verification; no model calls."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'UPLOAD_MANIFEST.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing/non-regular publication file: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Publication checksum mismatch: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    forbidden = ('datasets/', 'data/', 'outputs/', 'research/', 'assets/')
    for name in manifest['files_sha256']:
        if name.startswith(forbidden) or name.endswith(('.bin', '.safetensors', '.npy', '.npz', '.xlsx', '.zip', '.pdf')):
            raise ValueError('Forbidden publication artifact: ' + name)
    print(json.dumps({'inventory_files_verified': len(manifest['files_sha256']),
        'python_syntax_ok': True, 'dataset_and_model_artifacts_in_manifest': False}, indent=2))


if __name__ == '__main__':
    main()
