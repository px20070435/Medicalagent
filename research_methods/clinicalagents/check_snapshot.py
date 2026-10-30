"""Offline source audit. Does not import or execute upstream code."""
import ast
import hashlib
import json
from pathlib import Path


def audit():
    root = Path(__file__).resolve().parent
    upstream = root / 'upstream'
    manifest = json.loads((root / 'UPSTREAM_MANIFEST.json').read_text())
    expected = manifest['files_sha256']
    actual = {
        path.relative_to(upstream).as_posix()
        for path in upstream.rglob('*')
        if path.is_file() and '__pycache__' not in path.parts
    }
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    mismatches = []
    syntax_errors = []
    settings = set()
    module_imports = {}
    for name, digest in sorted(expected.items()):
        path = upstream / name
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(name)
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=name)
        except SyntaxError as exc:
            syntax_errors.append({'file': name, 'line': exc.lineno, 'message': exc.msg})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module_imports.setdefault(node.module, set()).add(name)
                if node.module == 'app_config':
                    settings.update(item.name for item in node.names)
    absent_modules = {}
    for name in ('app_config', 'experience_memory'):
        if name in module_imports and not (upstream / (name + '.py')).exists():
            absent_modules[name + '.py'] = sorted(module_imports[name])
    snapshot_ok = not (missing or unexpected or mismatches or syntax_errors)
    return {
        'commit': manifest['commit'],
        'files_checked': len(expected),
        'snapshot_ok': snapshot_ok,
        'missing_snapshot_files': missing,
        'unexpected_snapshot_files': unexpected,
        'sha256_mismatches': mismatches,
        'syntax_errors': syntax_errors,
        'missing_upstream_modules': absent_modules,
        'required_app_config_names': sorted(settings),
        'upstream_runnable': False,
        'runtime_status': 'not_validated; see README.md for integration blockers',
    }


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['snapshot_ok'] else 1)
