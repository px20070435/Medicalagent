"""Build a new immutable case corpus and the upstream-compatible workbook."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medagent.benchmarks.case_rag import CaseRetriever, build_corpus, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/merged_cases.json')
    parser.add_argument('--heldout', type=Path, default=ROOT / 'datasets/filtered_data_test_set.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = build_corpus(args.source, args.heldout, args.output_dir)
    retriever = CaseRetriever(args.output_dir)
    workbook = args.output_dir / 'MDDB.xlsx'
    retriever.export_workbook(workbook)
    manifest['workbook_sha256'] = sha(workbook)
    manifest['builder_code_sha256'] = sha(ROOT / 'medagent/benchmarks/case_rag.py')
    manifest['build_script_sha256'] = sha(Path(__file__))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
