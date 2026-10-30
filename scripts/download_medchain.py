"""Download only pinned public MedChain data; never fetch model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = '769a6de3333bf1ab5e283d2ccbdba62029686a07'
CODE_REVISION = '35ab53e3c601831a7bca181c72ecad6b12b55ff3'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Show sources without downloading')
    args = parser.parse_args()
    sources = json.loads((ROOT / 'configs/medchain-download.json').read_text())
    if args.plan:
        print(json.dumps(sources, ensure_ascii=False, indent=2))
        return
    from huggingface_hub import snapshot_download
    raw = ROOT / 'data/raw/medchain'
    raw.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id='ljwztc/MedChain', repo_type='dataset', revision=HF_REVISION,
        local_dir=raw, allow_patterns=['merged_cases.json', 'image_reports.json',
            'images_with_reports.txt', 'images.zip'])
    dataset = ROOT / 'datasets'
    dataset.mkdir(parents=True, exist_ok=True)
    for entry in sources['github_files']:
        target = dataset / entry['name']
        if target.exists():
            if digest(target) != entry['sha256']:
                raise ValueError('Existing file differs; refusing to replace: ' + str(target))
        else:
            temporary = target.with_suffix(target.suffix + '.part')
            with urlopen(entry['url'], timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if digest(temporary) != entry['sha256']:
                raise ValueError('Annotation checksum mismatch: ' + entry['name'])
            temporary.replace(target)
    extracted = ROOT / 'data/image_archive'
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / 'images.zip') as archive:
        for member in archive.infolist():
            target = extracted / member.filename
            if not target.resolve().is_relative_to(extracted.resolve()):
                raise ValueError('Unsafe archive path')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            temporary = target.with_suffix(target.suffix + '.extracting')
            with archive.open(member) as original, temporary.open('wb') as output:
                shutil.copyfileobj(original, output)
            temporary.replace(target)
    images = dataset / 'MedImg'
    images.mkdir(exist_ok=True)
    count = 0
    for path in sorted(extracted.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}:
            continue
        target = images / path.name
        if target.exists() or target.is_symlink():
            if digest(target) != digest(path):
                raise ValueError('Image basename collision: ' + path.name)
        else:
            target.symlink_to(path.resolve())
        count += 1
    for name in ['merged_cases.json', 'image_reports.json', 'images_with_reports.txt']:
        target = dataset / name
        original = raw / name
        if target.exists() or target.is_symlink():
            if digest(target) != digest(original):
                raise ValueError('Existing dataset differs: ' + name)
        else:
            target.symlink_to(original.resolve())
    report = dict(dataset='ljwztc/MedChain', revision=HF_REVISION, annotation_revision=CODE_REVISION,
        archive_image_files=count, files={name: digest(dataset / name)
            for name in ['filtered_data_test_set.json', 'merged_cases_room_structure.txt', 'merged_cases.json']})
    (ROOT / 'data/medchain-download-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
